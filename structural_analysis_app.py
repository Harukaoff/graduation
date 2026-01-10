import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import fem_lib
import draw_lib

st.set_page_config(
    layout="wide", 
    page_title="InstaStruct - 構造力学解析アプリ",
    page_icon="🏗️",
    menu_items={
        'Get Help': 'https://github.com/Harukaoff/graduation',
        'Report a bug': 'https://github.com/Harukaoff/graduation/issues',
        'About': """
        # InstaStruct
        
        手書き構造図から自動で構造解析を行うアプリです。
        
        **開発者**: 森本遥香 (DA22340)
        """
    }
)

# ==== 設定 ====
# Streamlit Cloud対応: 相対パスを優先、次に絶対パス、最後に環境変数
# 1. まず相対パスを試す（GitHubにアップロードした場合）
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

# 2. 相対パスが存在しない場合、元の絶対パスを試す（ローカル開発用）
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
if not os.path.exists(TEMPLATE_DIR):
    TEMPLATE_DIR = r"C:\Users\morim\Downloads\graduation\templates"

# 3. 環境変数で上書き可能（Streamlit Cloud用）
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_PATH)
TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", TEMPLATE_DIR)
TEMPLATE_FILES = {
    "pin": "pin.png",
    "roller": "roller.png",
    "fixed": "fixed.png",
    "beam": "beam.png",
    "load": "load.png",
    "momentl": "momentL.png",
    "momentr": "momentR.png",
    "udl": "UDL.png",
    "hinge": "hinge.png",
}
support_types = {"pin", "roller", "fixed", "hinge"}
load_types = {"load", "udl", "momentl", "momentr"}

def template_path(name):
    fname = TEMPLATE_FILES.get(name)
    return os.path.join(TEMPLATE_DIR, fname) if fname else None

def to_numpy(x):
    try: return x.cpu().numpy()
    except Exception: return np.array(x)

def order_cw_start_top_left(pts):
    pts = np.asarray(pts, float).reshape(-1, 2)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(-angles)
    pts_sorted = pts[order]
    miny = np.min(pts_sorted[:, 1])
    cand = np.where(np.isclose(pts_sorted[:, 1], miny, atol=1e-2))[0]
    idx = cand[np.argmin(pts_sorted[cand, 0])] if len(cand) > 1 else cand[0]
    pts_final = np.roll(pts_sorted, -idx, axis=0)
    return pts_final

def load_template_rgba(path):
    if not path or not os.path.exists(path): return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.ones_like(b) * 255
        img = cv2.merge([b, g, r, a])
    return img

def scale_image(img, scale):
    h, w = img.shape[:2]
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def rotate_image_keep_alpha(img, angle_deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    nw = int(h * abs_sin + w * abs_cos)
    nh = int(h * abs_cos + w * abs_sin)
    M[0, 2] += (nw / 2 - w / 2)
    M[1, 2] += (nh / 2 - h / 2)
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def overlay_rgba(base, overlay, center):
    bx, by = int(center[0]), int(center[1])
    oh, ow = overlay.shape[:2]
    x1 = bx - ow // 2
    y1 = by - oh // 2
    X1 = max(0, x1)
    X2 = min(base.shape[1], x1 + ow)
    Y1 = max(0, y1)
    Y2 = min(base.shape[0], y1 + oh)
    ox1 = X1 - x1
    oy1 = Y1 - y1
    ox2 = ox1 + (X2 - X1)
    oy2 = oy1 + (Y2 - Y1)
    crop = overlay[oy1:oy2, ox1:ox2]
    if crop.shape[2] < 4:
        base[Y1:Y2, X1:X2] = crop[..., :3]
        return base
    alpha = crop[..., 3:4] / 255.0
    for c in range(3):
        base[Y1:Y2, X1:X2, c] = (1.0 - alpha[..., 0]) * base[Y1:Y2, X1:X2, c] + alpha[..., 0] * crop[..., c]
    return base

def get_template_top_point(tpl):
    """テンプレート画像の最上端中央点を取得"""
    assert tpl is not None
    alpha = tpl[..., 3]
    pts = np.column_stack(np.where(alpha > 128))
    if len(pts) == 0:
        h, w = tpl.shape[:2]
        return np.array([w // 2, 0])
    # 最上端のy座標（画像座標系なので最小値）
    miny = np.min(pts[:, 0])
    # 最上端の点群
    top_pts = pts[pts[:, 0] == miny]
    # x座標の中央値を取得
    center_x = np.mean(top_pts[:, 1])
    top_pt = np.array([center_x, miny])
    return top_pt

def template_absolute_top(img_abs_center, template, angle=0):
    h, w = template.shape[:2]
    top_pt = get_template_top_point(template)
    offset = top_pt - np.array([w // 2, h // 2])
    theta = np.deg2rad(angle)
    rotM = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    rotated_offset = rotM @ offset
    return img_abs_center + rotated_offset

def align_nodes_y(nodes, thresh=8.0):
    """節点のy座標を揃える（x座標が近い場合はx座標も揃える）"""
    if len(nodes) == 0:
        return nodes
    
    xs = np.array([n[0] for n in nodes])
    ys = np.array([n[1] for n in nodes])
    used = np.zeros(len(nodes), dtype=bool)
    new_nodes = list(nodes)
    
    # まずy座標を揃える
    for i in range(len(nodes)):
        if used[i]: continue
        group = [i]
        for j in range(i + 1, len(nodes)):
            if abs(ys[i] - ys[j]) < thresh: 
                group.append(j)
        if len(group) > 1:
            avg_y = np.mean([ys[g] for g in group])
            for g in group:
                new_nodes[g] = np.array([new_nodes[g][0], avg_y])
                used[g] = True
        else:
            used[group[0]] = True
    
    # 次にx座標を揃える（垂直に並んでいる支点用）
    xs = np.array([n[0] for n in new_nodes])
    used = np.zeros(len(new_nodes), dtype=bool)
    for i in range(len(new_nodes)):
        if used[i]: continue
        group = [i]
        for j in range(i + 1, len(new_nodes)):
            if abs(xs[i] - xs[j]) < thresh:
                group.append(j)
        if len(group) > 1:
            avg_x = np.mean([xs[g] for g in group])
            for g in group:
                new_nodes[g] = np.array([avg_x, new_nodes[g][1]])
                used[g] = True
        else:
            used[group[0]] = True
    
    return new_nodes

def get_beam_endpoints(pts):
    """梁の四角形から最も離れた2点（端点）を取得"""
    dmax, pt1, pt2 = -1, None, None
    for i, p1 in enumerate(pts):
        for j, p2 in enumerate(pts):
            if i >= j:
                continue
            d = np.linalg.norm(p1 - p2)
            if d > dmax:
                dmax = d
                pt1 = p1
                pt2 = p2
    return pt1, pt2

def round_angle_deg(angle):
    return round(angle / 15) * 15

def find_nearest_node(pt, nodes):
    """最近傍節点のインデックスを返す"""
    if len(nodes) == 0:
        return -1
    dists = [np.linalg.norm(pt - n) for n in nodes]
    return int(np.argmin(dists))

def get_template_arrow_tip(tpl):
    """テンプレート画像内の矢じり先端のローカル座標を取得"""
    if tpl is None:
        return np.array([0, 0])
    
    alpha = tpl[..., 3]
    pts = np.column_stack(np.where(alpha > 128))
    
    if len(pts) == 0:
        h, w = tpl.shape[:2]
        return np.array([h // 2, w // 2])
    
    # テンプレートは下向き（90度）が基準
    # 矢じりは最下端（y最大）
    max_y = np.max(pts[:, 0])
    bottom_pts = pts[pts[:, 0] == max_y]
    center_x = np.mean(bottom_pts[:, 1])
    
    # (row, col) = (y, x) の順なので注意
    return np.array([max_y, center_x])

def get_rotated_arrow_tip(template, center, angle):
    """回転後のテンプレートの矢じり先端の絶対座標を計算
    
    Args:
        template: テンプレート画像
        center: テンプレートの中心座標（画像上の絶対座標 (x, y)）
        angle: 回転角度（度、0度=右向き、90度=下向き）
    
    Returns:
        矢じり先端の絶対座標 (x, y)
    """
    h, w = template.shape[:2]
    
    # テンプレート内の矢じり先端のローカル座標 (row, col)
    tip_local = get_template_arrow_tip(template)
    
    # テンプレート中心からのオフセット (row, col) = (dy, dx)
    offset_row = tip_local[0] - h // 2
    offset_col = tip_local[1] - w // 2
    
    # (x, y) 座標系に変換
    offset_x = offset_col
    offset_y = offset_row
    
    # 回転行列を適用（テンプレート基準角度90度を考慮）
    # テンプレートは90度（下向き）が基準なので、検出角度から90度を引く
    theta = np.deg2rad(angle - 90)
    
    # 標準的な回転行列（反時計回り）
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # オフセットを回転
    rotated_offset = rot_matrix @ np.array([offset_x, offset_y])
    
    # 絶対座標を計算
    tip_absolute = center + rotated_offset
    
    return tip_absolute

# タイトルとロゴ
col_logo, col_title = st.columns([1, 4])

with col_logo:
    # ロゴを表示
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
    else:
        st.write("🏗️")

with col_title:
    st.title("InstaStruct")
    st.write("構造力学解析アプリ")
    st.caption("手書き構造図から自動で構造解析を行い、変形図と応力図を出力します")

# サイドバー設定
with st.sidebar:
    # サイドバーにもロゴを表示
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    st.markdown("---")
    
    st.header("⚙️ 解析設定")
    
    # 検出精度向上のための設定
    st.subheader("🔍 検出精度設定")
    
    # 画像サイズ設定
    img_size = st.selectbox(
        "画像解析サイズ",
        [640, 800, 1024, 1280],
        index=0,
        help="大きいサイズほど小さな要素も検出しやすくなりますが、処理時間が長くなります"
    )
    
    # 検出後処理設定
    iou_threshold = st.slider(
        "重複除去閾値 (IoU)",
        0.1, 0.9, 0.45, 0.05,
        help="重複検出の除去基準。低いほど重複を厳しく除去します"
    )
    
    # 最大検出数
    max_det = st.slider(
        "最大検出数",
        100, 1000, 300, 50,
        help="1枚の画像で検出する要素の最大数"
    )
    
    st.markdown("---")
    
    # 信頼度設定方式の選択
    auto_conf = st.checkbox("🤖 自動信頼度調整", value=True, help="解析可能な構造が検出されるまで信頼度を自動調整します")
    
    if auto_conf:
        st.info("📊 自動調整モード: 解析可能な構造が検出されるまで信頼度を0.8から0.2まで自動調整します")
        conf_th = 0.8  # 初期値（自動調整で変更される）
    else:
        conf_th = st.slider("🎯 検出信頼度", 0.1, 0.9, 0.5, 0.05, 
                           help="値が高いほど確実な検出のみを採用します")
    
    # 画像前処理オプション
    st.subheader("🖼️ 画像前処理")
    
    enable_preprocessing = st.checkbox("画像前処理を有効にする", help="コントラスト調整や輪郭強調で検出精度を向上")
    
    if enable_preprocessing:
        # コントラスト調整
        contrast_factor = st.slider("コントラスト調整", 0.5, 2.0, 1.0, 0.1)
        
        # 輪郭強調
        edge_enhancement = st.checkbox("輪郭強調", value=False)
        
        # ノイズ除去
        noise_reduction = st.checkbox("ノイズ除去", value=False)
    
    st.markdown("---")
    
    # 固定値設定
    young = 2.0e2
    area = 9.0e2
    s_moment = 6.75e4
    load_value = 10.0
    moment_value = 10.0
    udl_value = 5.0
    
    st.markdown("---")
    st.subheader("📋 固定設定値")
    st.markdown(f"""
    **材料特性**
    - ヤング係数 E: `{young:.1e}`
    - 断面積 A: `{area:.1e}`
    - 断面二次モーメント I: `{s_moment:.1e}`
    
    **荷重設定**
    - 集中荷重: `{load_value:.1f}`
    - モーメント荷重: `{moment_value:.1f}`
    - 等分布荷重: `{udl_value:.1f}`
    """)

def preprocess_image(img, contrast_factor=1.0, edge_enhancement=False, noise_reduction=False):
    """画像前処理を実行"""
    processed_img = img.copy()
    
    # コントラスト調整
    if contrast_factor != 1.0:
        processed_img = cv2.convertScaleAbs(processed_img, alpha=contrast_factor, beta=0)
    
    # ノイズ除去
    if noise_reduction:
        processed_img = cv2.bilateralFilter(processed_img, 9, 75, 75)
    
    # 輪郭強調
    if edge_enhancement:
        # ガウシアンブラーを適用してからシャープニング
        blurred = cv2.GaussianBlur(processed_img, (3, 3), 0)
        processed_img = cv2.addWeighted(processed_img, 1.5, blurred, -0.5, 0)
    
    return processed_img

uploaded = st.file_uploader("📷 構造図画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("画像ファイルをアップロードしてください")
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 画像前処理を適用
if enable_preprocessing:
    img = preprocess_image(
        img, 
        contrast_factor=contrast_factor,
        edge_enhancement=edge_enhancement,
        noise_reduction=noise_reduction
    )
    # 前処理後の画像をPIL形式にも変換
    img_pil_processed = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
else:
    img_pil_processed = img_pil

col1, col2 = st.columns(2)
with col1:
    caption = "前処理後画像" if enable_preprocessing else "元画像"
    st.image(img_pil_processed, caption=caption, use_container_width=True)

TEMPL = {k: load_template_rgba(template_path(k)) for k in TEMPLATE_FILES}

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    st.error(f"モデルパスが存在しません: {MODEL_PATH}")
    st.stop()

if not st.button("🚀 解析実行", type="primary"):
    st.stop()

# 画像サイズを取得
img_height, img_width = img.shape[:2]

# 画像サイズに応じた閾値（画像の5%程度）
base_y_align_th = min(150.0, img_height * 0.05)
base_node_connect_th = min(100.0, max(img_width, img_height) * 0.03)  # 3%程度に縮小

def is_valid_structure(supports_count, beams_count, loads_count):
    """解析可能な構造かどうかを判定"""
    # 最低限の要素が必要
    if supports_count < 2:  # 支点が2つ以上
        return False
    if beams_count < 1:  # 梁が1つ以上
        return False
    if loads_count < 1:  # 荷重要素が1つ以上
        return False
    return True

# 画像認識実行
with st.spinner("画像認識中..."):
    model = YOLO(MODEL_PATH)
    
    # 自動信頼度調整
    if auto_conf:
        # 自動調整モード：解析可能な構造が見つかるまで信頼度を下げていく
        conf_candidates = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        best_conf = None
        best_result = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, test_conf in enumerate(conf_candidates):
            status_text.text(f"信頼度 {test_conf:.1f} で検出中...")
            progress_bar.progress((idx + 1) / len(conf_candidates))
            
            # テスト実行（改良された設定を使用）
            test_res = model(img, conf=test_conf, imgsz=img_size, iou=iou_threshold, max_det=max_det)[0]
            test_obb = test_res.obb
            
            # 要素数をカウント
            test_supports = 0
            test_beams = 0
            test_loads = 0
            
            if hasattr(test_obb, "xyxyxyxy"):
                test_N = len(to_numpy(test_obb.xyxyxyxy))
                for i in range(test_N):
                    test_conf_val = float(to_numpy(test_obb.conf[i]))
                    if test_conf_val < test_conf:
                        continue
                    test_cls_id = int(to_numpy(test_obb.cls[i]))
                    test_name = test_res.names[test_cls_id].lower().replace(" ", "")
                    
                    if test_name in support_types:
                        test_supports += 1
                    elif test_name == "beam":
                        test_beams += 1
                    elif test_name in load_types:
                        test_loads += 1
            
            # 解析可能な構造かチェック
            if is_valid_structure(test_supports, test_beams, test_loads):
                best_conf = test_conf
                best_result = test_res
                status_text.success(f"✅ 信頼度 {test_conf:.1f} で解析可能な構造を検出しました！")
                break
        
        progress_bar.empty()
        
        if best_conf is not None:
            conf_th = best_conf
            res = best_result
            st.info(f"🎯 自動調整結果: 信頼度 {conf_th:.1f} を使用")
        else:
            conf_th = 0.2
            res = model(img, conf=conf_th, imgsz=img_size, iou=iou_threshold, max_det=max_det)[0]
    else:
        # 手動モード（改良された設定を使用）
        res = model(img, conf=conf_th, imgsz=img_size, iou=iou_threshold, max_det=max_det)[0]
    
    # 固定閾値を使用
    y_align_th = base_y_align_th
    node_connect_th = base_node_connect_th

obb = res.obb
supports, beams, loads = [], [], []
N = len(to_numpy(obb.xyxyxyxy)) if hasattr(obb, "xyxyxyxy") else 0

for i in range(N):
    conf = float(to_numpy(obb.conf[i]))
    if conf < conf_th: continue
    cls_id = int(to_numpy(obb.cls[i]))
    name = res.names[cls_id].lower().replace(" ", "")
    pts = to_numpy(obb.xyxyxyxy[i]).reshape(4, 2)
    pts = order_cw_start_top_left(pts)
    # 角度計算
    if name in load_types:
        # 荷重の場合：短辺を基準とした矢印軸の計算
        # 4辺の長さを計算して短辺を特定
        edge_lengths = []
        for j in range(4):
            next_j = (j + 1) % 4
            length = np.linalg.norm(pts[next_j] - pts[j])
            edge_lengths.append((length, j, next_j))
        
        # 長さでソート（短い順）
        edge_lengths.sort()
        
        # 最も短い辺（短辺1）と2番目に短い辺（短辺2）を取得
        short_edge1 = edge_lengths[0]
        short_edge2 = edge_lengths[1]
        
        # 短辺1の中点
        p1_1 = pts[short_edge1[1]]
        p1_2 = pts[short_edge1[2]]
        short_midpoint1 = (p1_1 + p1_2) / 2
        
        # 短辺2の中点
        p2_1 = pts[short_edge2[1]]
        p2_2 = pts[short_edge2[2]]
        short_midpoint2 = (p2_1 + p2_2) / 2
        
        # 2つの短辺の中点を結んだ線が矢印の軸
        # まず仮の角度を計算（どちら向きかは後で梁との位置関係で決定）
        arrow_axis = short_midpoint2 - short_midpoint1
        angle_raw = math.degrees(math.atan2(arrow_axis[1], arrow_axis[0]))
        
        # 0-360度に正規化
        if angle_raw < 0:
            angle_raw += 360
        
        # 15度刻みに丸める
        angle = round_angle_deg(angle_raw)
        
        # 角度の候補は2つ（180度反対方向）
        # 後で梁との位置関係で正しい向きを決定するため、両方を保存
        angle_candidate1 = angle
        angle_candidate2 = (angle + 180) % 360
        
        # 矢じり位置の決定（梁に近い側の短辺中点）
        # 後で梁との距離を計算して決定するため、両方の中点を保存
        load_short_midpoints = (short_midpoint1, short_midpoint2)
    elif name == "beam":
        # 梁の場合：長辺の方向
        angle = round_angle_deg(math.degrees(math.atan2(pts[2][1] - pts[0][1], pts[2][0] - pts[0][0])))
    else:
        # 支点の場合
        angle = round_angle_deg(math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])))
    if name in support_types:
        tpl = TEMPL.get(name)
        node = None
        if tpl is not None:
            node = template_absolute_top(pts.mean(axis=0), tpl, angle)
        else:
            node = pts.mean(axis=0)
        supports.append(dict(type=name, node=node, pts=pts, angle=angle, conf=conf))
    elif name == "beam":
        beams.append({"type": "beam", "pts": pts, "angle": round_angle_deg(angle), "conf": conf})
    elif name in load_types:
        load_data = {"type": name, "pts": pts, "angle": round_angle_deg(angle), "conf": conf}
        # 短辺中点情報を追加
        if 'load_short_midpoints' in locals():
            load_data["short_midpoints"] = load_short_midpoints
        # 角度の候補を追加（180度反対方向も含む）
        if 'angle_candidate1' in locals() and 'angle_candidate2' in locals():
            load_data["angle_candidates"] = (angle_candidate1, angle_candidate2)
        loads.append(load_data)

nodes = np.array([s["node"] for s in supports]) if supports else np.empty((0, 2))
nodes = align_nodes_y(nodes, thresh=y_align_th) if len(nodes) >= 2 else nodes
for i, s in enumerate(supports): s["node"] = nodes[i]

# ===== 節点と梁の接続処理 =====
# 1. すべての節点を収集（支点 + 梁端点）
all_nodes = []
node_info = []  # 節点の情報（タイプ、元のインデックスなど）

# 支点の節点を追加
for i, s in enumerate(supports):
    all_nodes.append(s["node"])
    node_info.append({"type": "support", "support_idx": i, "support_type": s["type"]})

# ===== 重複梁の削除 =====
# バウンディングボックスが大きく重なっている梁を削除
def calculate_bbox_overlap(pts1, pts2):
    """2つのバウンディングボックスの重なり度合いを計算（0-1）"""
    # 各ボックスの範囲を計算
    x1_min, y1_min = np.min(pts1, axis=0)
    x1_max, y1_max = np.max(pts1, axis=0)
    x2_min, y2_min = np.min(pts2, axis=0)
    x2_max, y2_max = np.max(pts2, axis=0)
    
    # 重なり領域を計算
    overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    overlap_area = overlap_x * overlap_y
    
    # 各ボックスの面積
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # 小さい方の面積に対する重なり率
    if min(area1, area2) > 0:
        return overlap_area / min(area1, area2)
    return 0

beams_to_remove_duplicate = []
overlap_threshold = 0.3  # 30%以上重なっていたら重複とみなす

for i in range(len(beams)):
    if i in beams_to_remove_duplicate:
        continue
    
    for j in range(i + 1, len(beams)):
        if j in beams_to_remove_duplicate:
            continue
        
        # 重なり度合いを計算
        overlap = calculate_bbox_overlap(beams[i]["pts"], beams[j]["pts"])
        
        if overlap > overlap_threshold:
            # 信頼度の低い方を削除
            if beams[i]["conf"] < beams[j]["conf"]:
                beams_to_remove_duplicate.append(i)
                st.info(f"🗑️ 梁{i}を削除（梁{j}と{overlap*100:.1f}%重複、信頼度: {beams[i]['conf']:.2f} < {beams[j]['conf']:.2f}）")
                break
            else:
                beams_to_remove_duplicate.append(j)
                st.info(f"🗑️ 梁{j}を削除（梁{i}と{overlap*100:.1f}%重複、信頼度: {beams[j]['conf']:.2f} < {beams[i]['conf']:.2f}）")

# 重複梁を削除
if beams_to_remove_duplicate:
    beams = [b for i, b in enumerate(beams) if i not in beams_to_remove_duplicate]
    st.info(f"ℹ️ {len(beams_to_remove_duplicate)}本の重複梁を削除しました")

# 梁の端点を追加（まだスナップしていない状態）
beam_endpoints = []
for i, b in enumerate(beams):
    pt1, pt2 = get_beam_endpoints(b['pts'])
    beam_endpoints.append({
        "beam_idx": i,
        "pt1": pt1,
        "pt2": pt2,
        "angle": b["angle"],
        "conf": b["conf"]
    })

# 2. 全ての梁端点を収集
all_beam_endpoints = []
for be in beam_endpoints:
    all_beam_endpoints.append({
        "point": be["pt1"],
        "beam_idx": be["beam_idx"],
        "is_pt1": True
    })
    all_beam_endpoints.append({
        "point": be["pt2"],
        "beam_idx": be["beam_idx"],
        "is_pt1": False
    })

# 3. 特殊ケース：梁1つ、支点2つの場合は、梁の両端を2つの支点に接続
if len(beam_endpoints) == 1 and len(supports) == 2:
    
    # 2つの支点
    support0 = all_nodes[0]
    support1 = all_nodes[1]
    
    # 支点を結ぶ角度を計算
    support_vector = support1 - support0
    support_angle = math.degrees(math.atan2(support_vector[1], support_vector[0]))
    if support_angle < 0:
        support_angle += 360
    
    # 15度刻みに補正
    corrected_angle = round(support_angle / 15) * 15
    
    # 支点間の距離
    support_distance = np.linalg.norm(support_vector)
    
    # 補正後の角度で梁の端点を再計算
    angle_rad = math.radians(corrected_angle)
    beam_pt1 = support0  # 端点1は支点0
    beam_pt2 = support0 + support_distance * np.array([math.cos(angle_rad), math.sin(angle_rad)])  # 端点2は補正後の位置
    
    # 梁端点を更新
    beam_endpoints[0]["pt1"] = beam_pt1
    beam_endpoints[0]["pt2"] = beam_pt2
    beam_endpoints[0]["angle"] = corrected_angle
    
    # 各支点に最も近い梁端点を決定
    dist_s0_p1 = np.linalg.norm(support0 - beam_pt1)
    dist_s0_p2 = np.linalg.norm(support0 - beam_pt2)
    dist_s1_p1 = np.linalg.norm(support1 - beam_pt1)
    dist_s1_p2 = np.linalg.norm(support1 - beam_pt2)
    
    # 最適な組み合わせを選択
    if dist_s0_p1 + dist_s1_p2 < dist_s0_p2 + dist_s1_p1:
        # 支点0 -> 端点1、支点1 -> 端点2
        support_to_beam_connections = [
            {
                "support_idx": 0,
                "endpoint_idx": 0,  # pt1
                "distance": dist_s0_p1,
                "support_coord": support0.tolist() if isinstance(support0, np.ndarray) else support0,
                "endpoint_coord": beam_pt1.tolist() if isinstance(beam_pt1, np.ndarray) else beam_pt1
            },
            {
                "support_idx": 1,
                "endpoint_idx": 1,  # pt2
                "distance": dist_s1_p2,
                "support_coord": support1.tolist() if isinstance(support1, np.ndarray) else support1,
                "endpoint_coord": beam_pt2.tolist() if isinstance(beam_pt2, np.ndarray) else beam_pt2
            }
        ]
    else:
        # 支点0 -> 端点2、支点1 -> 端点1
        support_to_beam_connections = [
            {
                "support_idx": 0,
                "endpoint_idx": 1,  # pt2
                "distance": dist_s0_p2,
                "support_coord": support0.tolist() if isinstance(support0, np.ndarray) else support0,
                "endpoint_coord": beam_pt2.tolist() if isinstance(beam_pt2, np.ndarray) else beam_pt2
            },
            {
                "support_idx": 1,
                "endpoint_idx": 0,  # pt1
                "distance": dist_s1_p1,
                "support_coord": support1.tolist() if isinstance(support1, np.ndarray) else support1,
                "endpoint_coord": beam_pt1.tolist() if isinstance(beam_pt1, np.ndarray) else beam_pt1
            }
        ]
    
    # 各支点に最も近い梁端点を決定
else:
    # 通常ケース：支点から最も近い梁端点を探してクラスタに追加
    # 支点接続用の閾値を大きくする（通常の8倍）
    support_connect_th = node_connect_th * 8

    support_to_beam_connections = []
    for support_idx in range(len(supports)):
        support_node = all_nodes[support_idx]
        
        # 最も近い梁端点を探す
        min_dist = float('inf')
        closest_endpoint_idx = -1
        closest_endpoint_point = None
        
        for ep_idx, ep in enumerate(all_beam_endpoints):
            dist = np.linalg.norm(support_node - ep["point"])
            if dist < min_dist:
                min_dist = dist
                closest_endpoint_idx = ep_idx
                closest_endpoint_point = ep["point"]
        
        # 閾値内なら接続（より大きな閾値を使用）
        if min_dist < support_connect_th and closest_endpoint_idx >= 0:
            support_to_beam_connections.append({
                "support_idx": support_idx,
                "endpoint_idx": closest_endpoint_idx,
                "distance": min_dist,
                "support_coord": support_node.tolist() if isinstance(support_node, np.ndarray) else support_node,
                "endpoint_coord": closest_endpoint_point.tolist() if isinstance(closest_endpoint_point, np.ndarray) else closest_endpoint_point
            })
            # 接続成功
        else:
            # 接続できなかった場合
            pass

# 4. 支点に接続された端点のセットを作成
support_connected_endpoints = set()
for conn in support_to_beam_connections:
    support_connected_endpoints.add(conn["endpoint_idx"])

# 5. 近い端点同士をグループ化（クラスタリング）
# ただし、支点に接続された端点は独立したクラスタとして扱う
endpoint_clusters = []
used_endpoints = set()

# まず、支点に接続された端点を独立したクラスタとして追加
for conn in support_to_beam_connections:
    ep_idx = conn["endpoint_idx"]
    support_idx = conn["support_idx"]
    
    endpoint_clusters.append({
        "endpoints": [ep_idx],
        "connected_support": support_idx
    })
    used_endpoints.add(ep_idx)

# 次に、水平梁と鉛直梁の端点を優先的に接続
# 梁の角度を取得して水平・鉛直を判定
horizontal_beams = []  # 水平梁（0度、180度付近）
vertical_beams = []    # 鉛直梁（90度、270度付近）

for be_idx, be in enumerate(beam_endpoints):
    angle = be["angle"]
    # 15度の範囲で水平・鉛直を判定
    if (angle <= 15 or angle >= 345) or (165 <= angle <= 195):
        horizontal_beams.append(be_idx)
    elif (75 <= angle <= 105) or (255 <= angle <= 285):
        vertical_beams.append(be_idx)

# 水平梁と鉛直梁の端点を接続
hv_connections = []
for h_idx in horizontal_beams:
    h_ep1_idx = h_idx * 2
    h_ep2_idx = h_idx * 2 + 1
    
    for v_idx in vertical_beams:
        v_ep1_idx = v_idx * 2
        v_ep2_idx = v_idx * 2 + 1
        
        # 4つの組み合わせの距離を計算
        distances = [
            (h_ep1_idx, v_ep1_idx, np.linalg.norm(all_beam_endpoints[h_ep1_idx]["point"] - all_beam_endpoints[v_ep1_idx]["point"])),
            (h_ep1_idx, v_ep2_idx, np.linalg.norm(all_beam_endpoints[h_ep1_idx]["point"] - all_beam_endpoints[v_ep2_idx]["point"])),
            (h_ep2_idx, v_ep1_idx, np.linalg.norm(all_beam_endpoints[h_ep2_idx]["point"] - all_beam_endpoints[v_ep1_idx]["point"])),
            (h_ep2_idx, v_ep2_idx, np.linalg.norm(all_beam_endpoints[h_ep2_idx]["point"] - all_beam_endpoints[v_ep2_idx]["point"]))
        ]
        
        # 最も近い組み合わせを探す
        min_dist_combo = min(distances, key=lambda x: x[2])
        
        # 閾値内なら接続候補として記録
        if min_dist_combo[2] < node_connect_th * 1.5:  # 閾値を1.5倍に拡大
            hv_connections.append({
                "ep1_idx": min_dist_combo[0],
                "ep2_idx": min_dist_combo[1],
                "distance": min_dist_combo[2]
            })
            # 水平-鉛直接続成功

# 水平-鉛直接続をクラスタとして追加
for conn in hv_connections:
    ep1_idx = conn["ep1_idx"]
    ep2_idx = conn["ep2_idx"]
    
    # 既に使用されている端点はスキップ
    if ep1_idx in used_endpoints or ep2_idx in used_endpoints:
        continue
    
    # 支点に接続されている端点はスキップ
    if ep1_idx in support_connected_endpoints or ep2_idx in support_connected_endpoints:
        continue
    
    # クラスタを作成
    endpoint_clusters.append({
        "endpoints": [ep1_idx, ep2_idx],
        "connected_support": -1
    })
    used_endpoints.add(ep1_idx)
    used_endpoints.add(ep2_idx)

# 残りの端点同士をクラスタリング
for i, ep1 in enumerate(all_beam_endpoints):
    if i in used_endpoints:
        continue
    
    # 新しいクラスタを作成
    cluster = [i]
    used_endpoints.add(i)
    
    # 近い端点を探してクラスタに追加（支点に接続されていない端点のみ）
    for j, ep2 in enumerate(all_beam_endpoints):
        if j in used_endpoints:
            continue
        if j in support_connected_endpoints:
            continue
        
        # 距離による判定
        dist = np.linalg.norm(ep1["point"] - ep2["point"])
        
        # y座標の差による判定（水平方向の梁の接続用）
        y_diff = abs(ep1["point"][1] - ep2["point"][1])
        
        # 距離が閾値内、またはy座標の差が閾値内なら接続
        if dist < node_connect_th or y_diff < y_align_th:
            cluster.append(j)
            used_endpoints.add(j)
    
    # クラスタに接続された支点の情報を追加（この場合は-1）
    endpoint_clusters.append({
        "endpoints": cluster,
        "connected_support": -1
    })

# 6. 各クラスタの中心を節点として追加
beam_endpoint_to_node = {}  # 梁端点インデックス -> 節点インデックスのマッピング

for cluster_info in endpoint_clusters:
    cluster = cluster_info["endpoints"]
    connected_support = cluster_info["connected_support"]
    
    # クラスタ内の全端点の平均位置を計算
    cluster_points = [all_beam_endpoints[idx]["point"] for idx in cluster]
    cluster_center = np.mean(cluster_points, axis=0)
    
    # 支点に接続されている場合は、その支点を使用
    if connected_support >= 0:
        node_idx = connected_support
        node_coord = all_nodes[connected_support]
    else:
        # 既存の節点（支点）に近いかチェック
        min_dist_to_support = float('inf')
        snap_to_support_idx = -1
        
        for i in range(len(supports)):
            support_node = all_nodes[i]
            dist = np.linalg.norm(cluster_center - support_node)
            if dist < min_dist_to_support:
                min_dist_to_support = dist
                snap_to_support_idx = i
        
        # 支点に近い場合はスナップ
        if min_dist_to_support < node_connect_th and snap_to_support_idx >= 0:
            node_idx = snap_to_support_idx
            node_coord = all_nodes[snap_to_support_idx]
        else:
            # 新規節点として追加
            node_idx = len(all_nodes)
            node_coord = cluster_center
            all_nodes.append(cluster_center)
            node_info.append({"type": "beam_connection", "cluster_size": len(cluster)})
    
    # クラスタ内の全端点をこの節点にマッピング
    for ep_idx in cluster:
        beam_endpoint_to_node[ep_idx] = (node_idx, node_coord)

# 7. 梁の接続情報を作成
beam_connections = []
for be_idx, be in enumerate(beam_endpoints):
    # 端点1と端点2のインデックスを計算
    pt1_idx = be_idx * 2
    pt2_idx = be_idx * 2 + 1
    
    # マッピングから節点情報を取得
    node1_idx, node1_coord = beam_endpoint_to_node[pt1_idx]
    node2_idx, node2_coord = beam_endpoint_to_node[pt2_idx]
    
    # スナップ距離を計算
    min_dist1 = np.linalg.norm(be["pt1"] - node1_coord)
    min_dist2 = np.linalg.norm(be["pt2"] - node2_coord)
    
    # ===== 梁の角度を15度刻みに補正 + 垂直接続の検出 =====
    # 単純梁の場合は、既に補正済みの角度を使用
    if len(beam_endpoints) == 1 and len(supports) == 2:
        corrected_angle = be["angle"]  # 補正済みの角度
        current_angle = corrected_angle
    else:
        # 現在の角度を計算
        node1_arr = np.array(node1_coord) if not isinstance(node1_coord, np.ndarray) else node1_coord
        node2_arr = np.array(node2_coord) if not isinstance(node2_coord, np.ndarray) else node2_coord
        
        current_angle = math.degrees(math.atan2(node2_arr[1] - node1_arr[1], 
                                                node2_arr[0] - node1_arr[0]))
        if current_angle < 0:
            current_angle += 360
        
        # 15度刻みに丸める（角度のみ、座標は変更しない）
        corrected_angle = round(current_angle / 15) * 15
    
    # 垂直接続の検出: 他の梁と90度に近い場合は記録（座標は変更しない）
    is_perpendicular = False
    for existing_conn in beam_connections:
        existing_angle = existing_conn.get("angle", 0)
        angle_diff = abs(corrected_angle - existing_angle)
        
        # 角度差を0-180度の範囲に正規化
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 90度に近い場合（85-95度の範囲）
        if 85 <= angle_diff <= 95:
            is_perpendicular = True
            # 角度の補正は行うが、座標は変更しない
            if angle_diff < 90:
                corrected_angle = existing_angle + 90
            else:
                corrected_angle = existing_angle - 90
            
            # 0-360度の範囲に正規化
            corrected_angle = corrected_angle % 360
            break
    
    # バウンディングボックスの検出結果を優先するため、座標の補正は行わない
    # 角度のみ補正値を記録
    # （ラーメン構造では検出位置が重要なため、座標変更は避ける）
    
    beam_connections.append({
        "beam_idx": be["beam_idx"],
        "node1_idx": node1_idx,
        "node2_idx": node2_idx,
        "node1_coord": node1_coord.tolist() if isinstance(node1_coord, np.ndarray) else node1_coord,
        "node2_coord": node2_coord.tolist() if isinstance(node2_coord, np.ndarray) else node2_coord,
        "angle": corrected_angle,
        "original_angle": current_angle,
        "conf": be["conf"],
        "snap1_dist": min_dist1,
        "snap2_dist": min_dist2
    })

# ===== 重複梁の削除処理 =====
# 1つの支点から2本以上の梁が出ている場合、最も長い1本だけを残す
support_node_indices = [i for i, info in enumerate(node_info) if info.get("type") == "support"]

beams_to_remove = []
for support_idx in support_node_indices:
    # この支点に接続している梁を探す
    connected_beams = []
    for i, beam in enumerate(beam_connections):
        if beam["node1_idx"] == support_idx or beam["node2_idx"] == support_idx:
            # 梁の長さを計算
            node1 = np.array(beam["node1_coord"])
            node2 = np.array(beam["node2_coord"])
            length = np.linalg.norm(node2 - node1)
            connected_beams.append((i, length))
    
    # 2本以上接続している場合、最も長い1本以外を削除対象にする
    if len(connected_beams) > 1:
        # 長さでソート（降順）
        connected_beams.sort(key=lambda x: x[1], reverse=True)
        # 最も長い梁以外を削除対象に追加
        for beam_idx, _ in connected_beams[1:]:
            if beam_idx not in beams_to_remove:
                beams_to_remove.append(beam_idx)

# 削除対象の梁を除外
if beams_to_remove:
    beam_connections = [beam for i, beam in enumerate(beam_connections) if i not in beams_to_remove]

# ===== 梁のクロス検出と削除 =====
# 梁同士が交差している場合、片方を削除
def segments_intersect(p1, p2, p3, p4):
    """2つの線分が交差するかチェック（端点での接触は除く）"""
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    # 端点が同じ場合は交差とみなさない
    if np.allclose(p1, p3) or np.allclose(p1, p4) or np.allclose(p2, p3) or np.allclose(p2, p4):
        return False
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

beams_to_remove_cross = []
for i in range(len(beam_connections)):
    if i in beams_to_remove_cross:
        continue
    beam1 = beam_connections[i]
    p1 = np.array(beam1["node1_coord"])
    p2 = np.array(beam1["node2_coord"])
    
    for j in range(i + 1, len(beam_connections)):
        if j in beams_to_remove_cross:
            continue
        beam2 = beam_connections[j]
        p3 = np.array(beam2["node1_coord"])
        p4 = np.array(beam2["node2_coord"])
        
        # 交差チェック
        if segments_intersect(p1, p2, p3, p4):
            # 短い方の梁を削除対象にする
            len1 = np.linalg.norm(p2 - p1)
            len2 = np.linalg.norm(p4 - p3)
            if len1 < len2:
                beams_to_remove_cross.append(i)
            else:
                beams_to_remove_cross.append(j)
            break

# 交差している梁を削除
if beams_to_remove_cross:
    beam_connections = [beam for i, beam in enumerate(beam_connections) if i not in beams_to_remove_cross]
    st.info(f"ℹ️ 梁が交差していたため、{len(beams_to_remove_cross)}本の梁を削除しました")

# ===== 荷重の接続処理 =====
# 集中荷重・モーメント荷重の矢じり先端を梁上の節点に接続し、梁を分割
# 等分布荷重はボックスのx座標範囲で梁に作用
load_connections = []
beams_to_split = []  # 分割が必要な梁のリスト
udl_on_beams = []  # 等分布荷重が作用する梁のリスト

for l in loads:
    load_type = l["type"]
    angle = l["angle"]
    
    # 等分布荷重の場合は特別処理
    if load_type == "udl":
        # UDLボックスの情報を取得
        pts = l["pts"]
        x_min = np.min(pts[:, 0])
        x_max = np.max(pts[:, 0])
        y_min = np.min(pts[:, 1])
        y_max = np.max(pts[:, 1])
        box_center = pts.mean(axis=0)
        
        # バウンディングボックスの幅と高さを計算
        box_width = x_max - x_min
        box_height = y_max - y_min
        
        # 横長か縦長かを判定
        is_horizontal = box_width > box_height
        
        # バウンディングボックスの長辺の長さを計算
        # 4つの辺の長さを計算
        side_lengths = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            length = np.linalg.norm(p2 - p1)
            side_lengths.append(length)
        
        # 長辺の長さを取得
        udl_width = max(side_lengths)
        
        # 最も近い梁を見つけて角度を合わせる
        closest_beam = None
        min_dist_to_beam = float('inf')
        
        for beam in beam_connections:
            beam_a = np.array(beam["node1_coord"])
            beam_b = np.array(beam["node2_coord"])
            beam_center = (beam_a + beam_b) / 2
            dist_to_box = np.linalg.norm(beam_center - box_center)
            
            if dist_to_box < min_dist_to_beam:
                min_dist_to_beam = dist_to_box
                closest_beam = beam
        
        # 最も近い梁の角度に合わせる
        if closest_beam is not None:
            beam_angle = closest_beam["angle"]
            
            # 梁の方向ベクトルを計算
            beam_a = np.array(closest_beam["node1_coord"])
            beam_b = np.array(closest_beam["node2_coord"])
            beam_vector = beam_b - beam_a
            beam_length = np.linalg.norm(beam_vector)
            beam_unit = beam_vector / beam_length
            
            # 梁に垂直な方向ベクトルを計算
            beam_perp = np.array([-beam_vector[1], beam_vector[0]]) / beam_length  # 垂直ベクトル
            
            # 等分布荷重は梁に垂直な方向を向く
            # 梁の角度に90度を加えて垂直方向の角度を計算
            perp_angle = (beam_angle + 90) % 360
            
            # 荷重の向きを梁との位置関係とボックスの位置で決定
            beam_center = (beam_a + beam_b) / 2
            if box_center[1] < beam_center[1]:  # ボックスが梁より上
                # 下向き荷重（梁に向かって）
                angle = perp_angle
                load_direction = np.array([np.cos(np.deg2rad(perp_angle)), np.sin(np.deg2rad(perp_angle))])
            else:  # ボックスが梁より下
                # 上向き荷重（梁に向かって）
                angle = (perp_angle + 180) % 360
                load_direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            
            # 角度を15度刻みに丸める
            angle = round_angle_deg(angle)
            
            # バウンディングボックスの向きに応じて矢印を配置
            # 矢印の間隔を決定（約50ピクセル間隔）
            arrow_spacing = 50
            
            if is_horizontal:
                # 横長の場合：x方向に沿って配置（従来通り）
                num_arrows = max(3, int(udl_width / arrow_spacing))
                
                # 梁上でバウンディングボックスの範囲に対応する位置を計算
                t_center = np.dot(box_center - beam_a, beam_unit)
                half_width = udl_width / 2
                t_start = max(0, min(beam_length, t_center - half_width))
                t_end = max(0, min(beam_length, t_center + half_width))
                
                # 等間隔で矢印位置を計算
                udl_positions = []
                for i in range(num_arrows):
                    t = t_start + (t_end - t_start) * i / (num_arrows - 1) if num_arrows > 1 else t_center
                    pos = beam_a + t * beam_unit
                    udl_positions.append(pos)
            else:
                # 縦長の場合：y方向に沿って配置
                num_arrows = max(3, int(udl_width / arrow_spacing))
                
                # y方向の範囲で等間隔に配置
                udl_positions = []
                for i in range(num_arrows):
                    y_pos = y_min + (y_max - y_min) * i / (num_arrows - 1) if num_arrows > 1 else box_center[1]
                    # x座標は梁上の投影点
                    # y座標に対応する梁上の点を探す
                    # 簡易的に、box_centerのx座標を使用
                    pos = np.array([box_center[0], y_pos])
                    udl_positions.append(pos)
            
        else:
            # 梁が見つからない場合はボックス中心に配置
            udl_positions = [box_center]
            load_direction = np.array([0, 1])  # デフォルトは下向き
            angle = 90  # デフォルトは下向き
        
        # 最も近い梁に等分布荷重を適用
        # 梁の方向に沿って、バウンディングボックスの範囲に対応する部分を特定
        if closest_beam is not None:
            beam_idx = beam_connections.index(closest_beam)
            beam_a = np.array(closest_beam["node1_coord"])
            beam_b = np.array(closest_beam["node2_coord"])
            beam_vector = beam_b - beam_a
            beam_length = np.linalg.norm(beam_vector)
            beam_unit = beam_vector / beam_length
            
            # バウンディングボックスの範囲を梁上に投影
            if is_horizontal:
                # 横長の場合：t_start と t_end は既に計算済み
                pass
            else:
                # 縦長の場合：y方向の範囲を梁上に投影
                t_center = np.dot(box_center - beam_a, beam_unit)
                half_height = box_height / 2
                t_start = max(0, min(beam_length, t_center - half_height))
                t_end = max(0, min(beam_length, t_center + half_height))
            
            udl_on_beams.append({
                "beam_idx": beam_idx,
                "load_value": udl_value,
                "direction": load_direction,
                "angle": angle,
                "t_start": t_start / beam_length,  # 正規化された位置（0-1）
                "t_end": t_end / beam_length,      # 正規化された位置（0-1）
                "beam_length": beam_length
            })
        
        # 等分布荷重の接続情報を記録（表示用）
        # 複数の矢印位置を保存
        load_connections.append({
            "type": load_type,
            "tip_coord": udl_positions[0].tolist() if udl_positions else box_center.tolist(),
            "proj_coord": udl_positions[0].tolist() if udl_positions else box_center.tolist(),
            "node_idx": -1,
            "on_beam": udl_on_beams[-1]["beam_idx"] if udl_on_beams else -1,
            "beam_idx_in_list": -1,
            "beam_t": 0.5,
            "angle": angle,
            "is_horizontal": is_horizontal,  # 横長か縦長かのフラグ
            "conf": float(l["conf"]),
            "dist_to_beam": 0,
            "needs_split": False,
            "is_udl": True,
            "x_range": (x_min, x_max),
            "direction": load_direction.tolist(),
            "bbox_pts": pts.tolist(),  # バウンディングボックス座標を追加
            "bbox_center": box_center.tolist(),  # バウンディングボックス中心を追加
            "udl_arrow_positions": [pos.tolist() for pos in udl_positions],  # 複数の矢印位置
            "udl_width": udl_width,  # バウンディングボックスの長辺の長さ
            "closest_beam_angle": beam_angle if closest_beam else angle  # 梁の角度
        })
        continue
    
    # 集中荷重・モーメント荷重の処理
    # テンプレートの中心座標
    center = l["pts"].mean(axis=0)
    
    # 短辺中点情報がある場合は、梁に近い側を矢じり位置として使用
    if "short_midpoints" in l and load_type in ["load"]:
        midpoint1, midpoint2 = l["short_midpoints"]
        
        # まず仮の梁との距離を計算して、どちらの短辺中点が梁に近いかを判定
        min_dist_to_beam = float('inf')
        closest_midpoint = midpoint1
        
        # 全ての梁との距離を計算
        for beam in beam_connections:
            beam_a = np.array(beam["node1_coord"])
            beam_b = np.array(beam["node2_coord"])
            beam_center = (beam_a + beam_b) / 2
            
            dist1 = np.linalg.norm(midpoint1 - beam_center)
            dist2 = np.linalg.norm(midpoint2 - beam_center)
            
            if dist1 < min_dist_to_beam:
                min_dist_to_beam = dist1
                closest_midpoint = midpoint1
            if dist2 < min_dist_to_beam:
                min_dist_to_beam = dist2
                closest_midpoint = midpoint2
        
        # 梁に近い側の短辺中点を矢じり位置とする
        tip = closest_midpoint
    elif load_type in ["load"]:
        # フォールバック: テンプレートを使った計算
        tpl = TEMPL.get(load_type)
        if tpl is not None:
            tip = get_rotated_arrow_tip(tpl, center, angle)
        else:
            tip = center
    else:  # moment
        tip = center
    
    # 荷重の向きを梁との位置関係で決定
    # 角度候補がある場合は、梁に向かう方向を選択
    if "angle_candidates" in l:
        angle_candidate1, angle_candidate2 = l["angle_candidates"]
        
        # 最も近い梁を見つける
        bbox_center = l["pts"].mean(axis=0)
        closest_beam_center = None
        min_dist_to_beam = float('inf')
        
        for beam in beam_connections:
            beam_a = np.array(beam["node1_coord"])
            beam_b = np.array(beam["node2_coord"])
            beam_center = (beam_a + beam_b) / 2
            dist = np.linalg.norm(bbox_center - beam_center)
            
            if dist < min_dist_to_beam:
                min_dist_to_beam = dist
                closest_beam_center = beam_center
        
        if closest_beam_center is not None:
            # バウンディングボックスから梁への方向ベクトル
            to_beam = closest_beam_center - bbox_center
            to_beam_angle = math.degrees(math.atan2(to_beam[1], to_beam[0]))
            if to_beam_angle < 0:
                to_beam_angle += 360
            
            # 2つの角度候補のうち、梁に向かう方向に近い方を選択
            diff1 = abs(angle_candidate1 - to_beam_angle)
            diff2 = abs(angle_candidate2 - to_beam_angle)
            
            # 角度差を0-180度の範囲に正規化
            if diff1 > 180:
                diff1 = 360 - diff1
            if diff2 > 180:
                diff2 = 360 - diff2
            
            # より近い角度を選択
            if diff1 < diff2:
                angle = angle_candidate1
            else:
                angle = angle_candidate2
    
    # 角度から方向ベクトルを計算
    angle_rad = np.deg2rad(angle)
    load_direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    # 最も近い梁を探して、梁上に投影
    best_beam = None
    best_beam_idx = -1
    best_proj = None
    best_dist = 1e9
    best_t = 0.0
    
    for idx, beam in enumerate(beam_connections):
        a = np.array(beam["node1_coord"])
        b = np.array(beam["node2_coord"])
        ba = b - a
        denom = np.dot(ba, ba) + 1e-12
        t = np.dot(tip - a, ba) / denom
        t = max(0.0, min(1.0, t))
        proj = a + t * ba
        dist = np.linalg.norm(tip - proj)
        if dist < best_dist:
            best_dist = dist
            best_beam = beam
            best_beam_idx = idx
            best_proj = proj
            best_t = t
    
    # 投影点を梁の4等分点の最近傍にスナップ
    if best_proj is not None and best_beam is not None:
        a = np.array(best_beam["node1_coord"])
        b = np.array(best_beam["node2_coord"])
        
        # 梁を4等分する点（t = 0, 0.25, 0.5, 0.75, 1.0）
        quarter_points = [a + t * (b - a) for t in [0.0, 0.25, 0.5, 0.75, 1.0]]
        quarter_t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # 投影点から最も近い4等分点を探す
        min_dist_to_quarter = float('inf')
        best_quarter_idx = 2  # デフォルトは中点
        for i, qp in enumerate(quarter_points):
            dist = np.linalg.norm(best_proj - qp)
            if dist < min_dist_to_quarter:
                min_dist_to_quarter = dist
                best_quarter_idx = i
        
        # 最近傍の4等分点を使用
        snapped_proj = quarter_points[best_quarter_idx]
        snapped_t = quarter_t_values[best_quarter_idx]
        
        # 既存節点との距離をチェック
        min_dist_to_node = float('inf')
        snap_node_idx = -1
        for i, node in enumerate(all_nodes):
            dist = np.linalg.norm(snapped_proj - node)
            if dist < min_dist_to_node:
                min_dist_to_node = dist
                snap_node_idx = i
        
        # 梁の端点（既存節点）に近い場合はスナップ
        if min_dist_to_node < 10.0 and snap_node_idx >= 0:
            load_node_idx = snap_node_idx
            load_node_coord = all_nodes[snap_node_idx]
            needs_split = False
            final_t = snapped_t
        else:
            # 梁の途中に新規節点を追加
            load_node_idx = len(all_nodes)
            load_node_coord = snapped_proj
            all_nodes.append(snapped_proj)
            node_info.append({"type": "load_point", "load_type": load_type})
            needs_split = True
            final_t = snapped_t
            
            # 梁の分割が必要（tが0.1～0.9の範囲、つまり端点から十分離れている場合）
            if 0.1 < snapped_t < 0.9:
                beams_to_split.append({
                    "beam_idx": best_beam_idx,
                    "split_node_idx": load_node_idx,
                    "split_t": snapped_t,
                    "original_beam": best_beam
                })
        
        # 投影点を更新
        best_proj = snapped_proj
        best_t = final_t
    else:
        load_node_idx = -1
        load_node_coord = tip
        needs_split = False
    
    load_connections.append({
        "type": load_type,
        "tip_coord": tip.tolist(),
        "proj_coord": best_proj.tolist() if best_proj is not None else tip.tolist(),
        "node_idx": load_node_idx,
        "on_beam": best_beam["beam_idx"] if best_beam else -1,
        "beam_idx_in_list": best_beam_idx,
        "beam_t": best_t,
        "angle": angle,
        "conf": float(l["conf"]),
        "dist_to_beam": best_dist,
        "needs_split": needs_split,
        "is_udl": False,
        "direction": load_direction.tolist(),
        "bbox_pts": l["pts"].tolist(),  # バウンディングボックス座標を追加
        "bbox_center": center.tolist()  # バウンディングボックス中心を追加
    })

# ===== 等分布荷重による梁の分割処理 =====
# 等分布荷重が作用する範囲で梁を分割
udl_beams_to_split = []
for udl in udl_on_beams:
    beam_idx = udl["beam_idx"]
    t_start = udl["t_start"]
    t_end = udl["t_end"]
    
    # 範囲の始点と終点で梁を分割
    if beam_idx < len(beam_connections):
        beam = beam_connections[beam_idx]
        beam_a = np.array(beam["node1_coord"])
        beam_b = np.array(beam["node2_coord"])
        
        # 始点の節点を追加（t_startが0より大きい場合）
        if t_start > 0.05:  # 端点から十分離れている場合のみ
            split_coord_start = beam_a + t_start * (beam_b - beam_a)
            split_node_idx_start = len(all_nodes)
            all_nodes.append(split_coord_start)
            node_info.append({"type": "udl_boundary", "udl_idx": len(udl_on_beams) - 1})
            
            udl_beams_to_split.append({
                "beam_idx": beam_idx,
                "split_node_idx": split_node_idx_start,
                "split_t": t_start,
                "original_beam": beam
            })
        
        # 終点の節点を追加（t_endが1より小さい場合）
        if t_end < 0.95:  # 端点から十分離れている場合のみ
            split_coord_end = beam_a + t_end * (beam_b - beam_a)
            split_node_idx_end = len(all_nodes)
            all_nodes.append(split_coord_end)
            node_info.append({"type": "udl_boundary", "udl_idx": len(udl_on_beams) - 1})
            
            udl_beams_to_split.append({
                "beam_idx": beam_idx,
                "split_node_idx": split_node_idx_end,
                "split_t": t_end,
                "original_beam": beam
            })
        
        # UDL情報に分割後の範囲情報を追加
        udl["split_t_start"] = t_start
        udl["split_t_end"] = t_end

# 集中荷重と等分布荷重の分割を統合
beams_to_split.extend(udl_beams_to_split)

# ===== 梁の分割処理 =====
# 荷重が作用している位置で梁を2つに分割
if beams_to_split:
    # 分割する梁をインデックスの降順でソート（後ろから処理）
    beams_to_split.sort(key=lambda x: x["beam_idx"], reverse=True)
    
    new_beam_connections = []
    for i, beam in enumerate(beam_connections):
        # この梁が分割対象か確認
        splits_for_this_beam = [s for s in beams_to_split if s["beam_idx"] == i]
        
        if splits_for_this_beam:
            # 分割点をt値でソート
            splits_for_this_beam.sort(key=lambda x: x["split_t"])
            
            # 元の梁の情報
            node1_idx = beam["node1_idx"]
            node1_coord = np.array(beam["node1_coord"])
            node2_idx = beam["node2_idx"]
            node2_coord = np.array(beam["node2_coord"])
            
            # 分割点ごとに新しい梁を作成
            current_node_idx = node1_idx
            current_coord = node1_coord
            
            for split in splits_for_this_beam:
                split_node_idx = split["split_node_idx"]
                split_coord = np.array(all_nodes[split_node_idx])
                
                # 分割された梁の前半部分
                new_beam_connections.append({
                    "beam_idx": beam["beam_idx"],
                    "node1_idx": current_node_idx,
                    "node2_idx": split_node_idx,
                    "node1_coord": current_coord.tolist() if isinstance(current_coord, np.ndarray) else current_coord,
                    "node2_coord": split_coord.tolist(),
                    "angle": beam["angle"],
                    "original_angle": beam.get("original_angle", beam["angle"]),
                    "conf": beam["conf"],
                    "snap1_dist": 0.0,
                    "snap2_dist": 0.0,
                    "is_split": True
                })
                
                current_node_idx = split_node_idx
                current_coord = split_coord
            
            # 最後の部分（分割点から端点2まで）
            new_beam_connections.append({
                "beam_idx": beam["beam_idx"],
                "node1_idx": current_node_idx,
                "node2_idx": node2_idx,
                "node1_coord": current_coord.tolist() if isinstance(current_coord, np.ndarray) else current_coord,
                "node2_coord": node2_coord.tolist() if isinstance(node2_coord, np.ndarray) else node2_coord,
                "angle": beam["angle"],
                "original_angle": beam.get("original_angle", beam["angle"]),
                "conf": beam["conf"],
                "snap1_dist": 0.0,
                "snap2_dist": 0.0,
                "is_split": True
            })
        else:
            # 分割不要な梁はそのまま追加
            new_beam_connections.append(beam)
    
    # 梁のリストを更新
    beam_connections = new_beam_connections

# ===== バウンディングボックス表示画像生成 =====
bbox_img = img.copy()

# 色の定義（BGR形式）
colors = {
    'support': (0, 0, 255),    # 赤：支点
    'beam': (255, 0, 0),       # 青：梁
    'load': (0, 255, 0),       # 緑：荷重
    'udl': (0, 255, 255),      # 黄：等分布荷重
    'momentl': (255, 0, 255),  # マゼンタ：左モーメント
    'momentr': (255, 255, 0)   # シアン：右モーメント
}

# 検出された全要素のバウンディングボックスを描画
for support in supports:
    pts = support["pts"].astype(int)
    color = colors.get('support', (128, 128, 128))
    cv2.polylines(bbox_img, [pts], True, color, 3)
    # ラベルを描画
    center = pts.mean(axis=0).astype(int)
    cv2.putText(bbox_img, f"{support['type']}", (center[0]-20, center[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # 信頼度を表示
    cv2.putText(bbox_img, f"{support['conf']:.2f}", (center[0]-20, center[1]+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

for beam in beams:
    pts = beam["pts"].astype(int)
    color = colors.get('beam', (128, 128, 128))
    cv2.polylines(bbox_img, [pts], True, color, 3)
    # ラベルを描画
    center = pts.mean(axis=0).astype(int)
    cv2.putText(bbox_img, "beam", (center[0]-20, center[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # 信頼度を表示
    cv2.putText(bbox_img, f"{beam['conf']:.2f}", (center[0]-20, center[1]+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

for load in loads:
    pts = load["pts"].astype(int)
    load_type = load["type"]
    color = colors.get(load_type, (128, 128, 128))
    cv2.polylines(bbox_img, [pts], True, color, 3)
    # ラベルを描画
    center = pts.mean(axis=0).astype(int)
    cv2.putText(bbox_img, load_type, (center[0]-20, center[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # 信頼度と角度を表示
    cv2.putText(bbox_img, f"{load['conf']:.2f}", (center[0]-20, center[1]+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(bbox_img, f"{load['angle']:.0f}°", (center[0]-20, center[1]+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

with col1:
    st.image(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB), caption="検出結果", use_container_width=True)

with col2:
    # ===== 清書画像生成 =====
    cleaned = np.ones_like(img) * 255

    # 梁を描画（接続後の節点座標を使用、15度刻みで補正）
    for conn in beam_connections:
        # 接続後の節点座標を取得
        node1_idx = conn["node1_idx"]
        node2_idx = conn["node2_idx"]
        pt1 = np.array(all_nodes[node1_idx])
        pt2 = np.array(all_nodes[node2_idx])
        
        # 15度刻みに角度を補正
        vector = pt2 - pt1
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        corrected_angle = round(angle / 15) * 15
        
        # 補正後の座標を計算
        length = np.linalg.norm(vector)
        angle_rad = math.radians(corrected_angle)
        pt2_corrected = pt1 + length * np.array([math.cos(angle_rad), math.sin(angle_rad)])
        
        cv2.line(cleaned, tuple(map(int, pt1)), tuple(map(int, pt2_corrected)), (80, 80, 80), 8)

# 支点を描画（テンプレート上端が節点位置になるように配置）
for i, s in enumerate(supports):
    name = s["type"]
    tpl = TEMPL.get(name)
    original_angle = s["angle"]
    
    # 支点の角度を調整
    if name in ["pin", "roller"]:
        # ピン支点とピンローラー支点は常に0度（角度固定）
        angle = 0
    elif name == "fixed":
        # 固定支点は90度回転
        angle = original_angle + 90
    else:
        # その他（ヒンジなど）は元の角度
        angle = original_angle
    
    if tpl is not None:
        tpl_scaled = scale_image(tpl, 0.8)
        tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle)
        
        # テンプレート上端の位置を計算
        h, w = tpl_rot.shape[:2]
        top_pt_local = get_template_top_point(tpl_rot)
        
        # 節点位置からテンプレート上端へのオフセットを計算
        # 節点位置 = テンプレート上端になるように配置
        support_node_idx = node_info[i]["support_idx"] if i < len(node_info) and node_info[i]["type"] == "support" else None
        if support_node_idx is not None:
            # 支点の実際の節点座標を取得
            for j, info in enumerate(node_info):
                if info.get("type") == "support" and info.get("support_idx") == support_node_idx:
                    node_coord = np.array(all_nodes[j])
                    # テンプレート中心位置を計算（上端が節点位置になるように）
                    center_offset = np.array([w // 2, h // 2]) - top_pt_local
                    center = node_coord + center_offset
                    cleaned = overlay_rgba(cleaned, tpl_rot, center)
                    break
        else:
            # フォールバック: 元の方法
            center = s["node"]
            cleaned = overlay_rgba(cleaned, tpl_rot, center)

# すべての節点を描画
for i, node in enumerate(all_nodes):
    node_coord = node if isinstance(node, np.ndarray) else np.array(node)
    info = node_info[i] if i < len(node_info) else {"type": "unknown"}
    
    if info["type"] == "support":
        # 支点節点（赤）
        cv2.circle(cleaned, tuple(map(int, node_coord)), 10, (0, 0, 255), 2)
        cv2.putText(cleaned, f"N{i}", (int(node_coord[0]) + 12, int(node_coord[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif info["type"] == "beam_endpoint":
        # 梁端点（青）
        cv2.circle(cleaned, tuple(map(int, node_coord)), 8, (255, 0, 0), 2)
        cv2.putText(cleaned, f"N{i}", (int(node_coord[0]) + 12, int(node_coord[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    elif info["type"] == "load_point":
        # 荷重作用点（緑）
        cv2.circle(cleaned, tuple(map(int, node_coord)), 8, (0, 200, 0), 2)
        cv2.putText(cleaned, f"N{i}", (int(node_coord[0]) + 12, int(node_coord[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

# 荷重を描画（矢じり先端と梁上の点を接続）
for l in load_connections:
    name = l["type"]
    tpl = TEMPL.get(name)
    angle = l["angle"]
    
    # 矢じり先端の座標（回転後の計算済み座標）
    tip = np.array(l["tip_coord"])
    
    # 梁上の接続点座標（4等分点）
    node_idx = l["node_idx"]
    if node_idx >= 0 and node_idx < len(all_nodes):
        proj = np.array(all_nodes[node_idx])
    else:
        proj = np.array(l["proj_coord"])
    
    # 等分布荷重の場合は複数の矢印を描画
    if l.get("is_udl", False) and "udl_arrow_positions" in l:
        # 複数の矢印位置に円を描画
        for pos in l["udl_arrow_positions"]:
            pos_array = np.array(pos)
            cv2.circle(cleaned, tuple(map(int, pos_array)), 5, (0, 0, 255), -1)
    elif not l.get("is_udl", False):
        # 集中荷重の場合、矢じり先端と梁上の点を線で接続
        # 接続線を描画（緑色の細線）
        cv2.line(cleaned, tuple(map(int, tip)), tuple(map(int, proj)), (0, 200, 0), 3)
        
        # 矢じり先端に小さな円を描画（荷重の作用点）
        cv2.circle(cleaned, tuple(map(int, tip)), 5, (0, 0, 255), -1)
        
        # 梁上の接続点に円を描画
        cv2.circle(cleaned, tuple(map(int, proj)), 6, (255, 0, 0), 2)
    
    # 荷重テンプレートを配置
    if tpl is not None and "bbox_pts" in l:
        # 等分布荷重の場合は複数の矢印を配置
        if l.get("is_udl", False) and "udl_arrow_positions" in l:
            # 通常サイズのテンプレートを使用（個々の矢印は標準サイズ）
            tpl_scaled = scale_image(tpl, 0.6)
            
            # 横長・縦長に応じてテンプレートの回転角度を調整
            is_horizontal = l.get("is_horizontal", True)
            
            if is_horizontal:
                # 横長の場合：梁に垂直に表示（従来通り）
                template_rotation = angle + 90
            else:
                # 縦長の場合：さらに90度回転
                template_rotation = angle + 180
            
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, template_rotation)
            
            # 矢じりを矢印位置に配置
            h_rot, w_rot = tpl_rot.shape[:2]
            tip_local_rot = get_template_arrow_tip(tpl_rot)
            
            # テンプレート中心からのオフセット
            offset_row = tip_local_rot[0] - h_rot // 2
            offset_col = tip_local_rot[1] - w_rot // 2
            offset_x = offset_col
            offset_y = offset_row
            
            # 複数の矢印位置に対してテンプレートを配置
            for arrow_pos in l["udl_arrow_positions"]:
                arrow_pos_array = np.array(arrow_pos)
                
                # 角度に応じてオフセットの符号を調整
                # 下向き（45-135度）の場合はマイナス、それ以外はプラス
                if 45 <= angle <= 135:  # 下向き矢印
                    template_center = arrow_pos_array - np.array([offset_x, offset_y])
                else:  # その他の角度
                    template_center = arrow_pos_array + np.array([offset_x, offset_y])
                
                cleaned = overlay_rgba(cleaned, tpl_rot, template_center)
        
        # 集中荷重の場合の処理
        elif not l.get("is_udl", False):
            # 対応する荷重データから短辺中点情報を取得
            load_data = None
            bbox_center = np.array(l["bbox_center"])
            
            # 荷重データから短辺中点を探す
            for load in loads:
                if "short_midpoints" in load:
                    load_center = load["pts"].mean(axis=0)
                    if np.linalg.norm(load_center - bbox_center) < 20:  # 中心が近い荷重を探す
                        load_data = load
                        break
            
            if load_data is not None and "short_midpoints" in load_data:
                midpoint1, midpoint2 = load_data["short_midpoints"]
                
                # 矢印軸（短辺中点を結ぶ線）の計算
                arrow_axis = midpoint2 - midpoint1
                axis_length = np.linalg.norm(arrow_axis)
                axis_center = (midpoint1 + midpoint2) / 2
                
                # 矢印軸の角度（梁との位置関係で修正された角度を使用）
                axis_angle = angle  # 既に梁との位置関係で修正済みの角度
                
                # テンプレートのスケール（軸の長さに合わせる）
                tpl_h, tpl_w = tpl.shape[:2]
                # テンプレートは横向き（幅が軸の長さに対応）
                scale = (axis_length / tpl_w) * 0.9  # 少し小さめに調整
                
                tpl_scaled = scale_image(tpl, scale)
                
                # テンプレートを軸の角度に回転
                # テンプレートは下向き（90度）が基準なので、角度を調整
                # 90度 → -90度回転、270度 → 90度回転、0度 → -180度回転、180度 → 0度回転
                template_rotation = axis_angle - 180
                tpl_rot = rotate_image_keep_alpha(tpl_scaled, template_rotation)
                
                # 矢じりを梁上の接続点に配置
                # 回転後のテンプレート内の矢じり位置を取得
                h_rot, w_rot = tpl_rot.shape[:2]
                tip_local_rot = get_template_arrow_tip(tpl_rot)
                
                # テンプレート中心からのオフセット (row, col)
                offset_row = tip_local_rot[0] - h_rot // 2
                offset_col = tip_local_rot[1] - w_rot // 2
                
                # (x, y) 座標系に変換
                offset_x = offset_col
                offset_y = offset_row
                
                # 梁上の接続点を取得
                proj_coord = np.array(l["proj_coord"])
                
                # テンプレート中心位置を計算（矢じりが梁上の接続点に来るように）
                # 角度に応じてオフセットの符号を調整
                if axis_angle == 90:  # 下向き矢印
                    template_center = proj_coord - np.array([offset_x, offset_y])
                else:  # その他の角度（上向き、左右など）
                    template_center = proj_coord + np.array([offset_x, offset_y])
                
                cleaned = overlay_rgba(cleaned, tpl_rot, template_center)
            else:
                # 短辺中点情報がない場合のフォールバック
                bbox_pts = np.array(l["bbox_pts"])
                bbox_center = np.array(l["bbox_center"])
                
                # バウンディングボックスのサイズに合わせてスケール
                bbox_width = np.max(bbox_pts[:, 0]) - np.min(bbox_pts[:, 0])
                bbox_height = np.max(bbox_pts[:, 1]) - np.min(bbox_pts[:, 1])
                
                tpl_h, tpl_w = tpl.shape[:2]
                scale_x = bbox_width / tpl_w
                scale_y = bbox_height / tpl_h
                scale = min(scale_x, scale_y) * 0.8
                
                tpl_scaled = scale_image(tpl, scale)
                # テンプレートは下向き（90度）が基準なので、角度を調整
                template_rotation = angle - 180
                tpl_rot = rotate_image_keep_alpha(tpl_scaled, template_rotation)
                
                # 矢じりを梁上の接続点に配置
                h_rot, w_rot = tpl_rot.shape[:2]
                tip_local_rot = get_template_arrow_tip(tpl_rot)
                
                # テンプレート中心からのオフセット (row, col)
                offset_row = tip_local_rot[0] - h_rot // 2
                offset_col = tip_local_rot[1] - w_rot // 2
                
                # (x, y) 座標系に変換
                offset_x = offset_col
                offset_y = offset_row
                
                # 梁上の接続点を取得
                proj_coord = np.array(l["proj_coord"])
                
                # テンプレート中心位置を計算（矢じりが梁上の接続点に来るように）
                # 角度に応じてオフセットの符号を調整
                if angle == 90:  # 下向き矢印
                    template_center = proj_coord - np.array([offset_x, offset_y])
                else:  # その他の角度（上向き、左右など）
                    template_center = proj_coord + np.array([offset_x, offset_y])
                
                cleaned = overlay_rgba(cleaned, tpl_rot, template_center)
            
            cleaned = overlay_rgba(cleaned, tpl_rot, template_center)
    elif tpl is not None:
        # フォールバック: 従来の方法
        tpl_scaled = scale_image(tpl, 0.9)
        # テンプレートは下向き（90度）が基準なので、角度を調整
        template_rotation = angle - 180
        tpl_rot = rotate_image_keep_alpha(tpl_scaled, template_rotation)
        
        # 回転後のテンプレート内の矢じり位置を取得
        h_rot, w_rot = tpl_rot.shape[:2]
        tip_local_rot = get_template_arrow_tip(tpl_rot)
        
        # テンプレート中心からのオフセット (row, col)
        offset_row = tip_local_rot[0] - h_rot // 2
        offset_col = tip_local_rot[1] - w_rot // 2
        
        # (x, y) 座標系に変換
        offset_x = offset_col
        offset_y = offset_row
        
        # テンプレート中心位置を計算（矢じりが梁上の接続点に来るように）
        proj_coord = np.array(l["proj_coord"])
        # 角度に応じてオフセットの符号を調整
        if angle == 90:  # 下向き矢印
            template_center = proj_coord - np.array([offset_x, offset_y])
        else:  # その他の角度（上向き、左右など）
            template_center = proj_coord + np.array([offset_x, offset_y])
        
        cleaned = overlay_rgba(cleaned, tpl_rot, template_center)

with col2:
    st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), "清書画像", use_container_width=True)

st.success("✅ 画像認識・清書完了")

# ===== FEM解析用データ構造への変換 =====
with st.spinner("FEM解析データ準備中..."):
    # ===== 孤立節点の削除 =====
    # 梁に接続されている節点のインデックスを収集
    connected_nodes = set()
    for conn in beam_connections:
        connected_nodes.add(conn["node1_idx"])
        connected_nodes.add(conn["node2_idx"])
    
    # 孤立節点（梁に接続されていない節点）を特定
    isolated_nodes = []
    for i in range(len(all_nodes)):
        if i not in connected_nodes:
            isolated_nodes.append(i)
    
    # 孤立節点を削除し、節点インデックスを再マッピング
    if isolated_nodes:
        st.info(f"ℹ️ 孤立節点を{len(isolated_nodes)}個削除しました")
        
        # 新しい節点リストと情報リストを作成
        new_all_nodes = []
        new_node_info = []
        old_to_new_idx = {}  # 古いインデックス → 新しいインデックスのマッピング
        
        new_idx = 0
        for old_idx in range(len(all_nodes)):
            if old_idx not in isolated_nodes:
                new_all_nodes.append(all_nodes[old_idx])
                new_node_info.append(node_info[old_idx])
                old_to_new_idx[old_idx] = new_idx
                new_idx += 1
        
        # 梁の接続情報を更新
        for conn in beam_connections:
            conn["node1_idx"] = old_to_new_idx[conn["node1_idx"]]
            conn["node2_idx"] = old_to_new_idx[conn["node2_idx"]]
        
        # 荷重の接続情報を更新
        for l in load_connections:
            if l["node_idx"] >= 0 and l["node_idx"] in old_to_new_idx:
                l["node_idx"] = old_to_new_idx[l["node_idx"]]
            else:
                l["node_idx"] = -1  # 孤立節点に接続していた荷重は無効化
        
        # リストを更新
        all_nodes = new_all_nodes
        node_info = new_node_info
    
    # all_nodesを使用（孤立節点削除済み）
    num_nodes = len(all_nodes)
    
    # nodes_df作成
    nodes_df = pd.DataFrame(columns=['x', 'y', 'rc_x', 'rc_y', 'rc_m', 'ef_x', 'ef_y', 'ef_m'])
    for i, node in enumerate(all_nodes):
        node_coord = node if isinstance(node, np.ndarray) else np.array(node)
        nodes_df.loc[i] = [float(node_coord[0]), float(node_coord[1]), 0, 0, 0, 0, 0, 0]
    
    # 拘束条件設定（node_infoを使用）
    for i, info in enumerate(node_info):
        if info["type"] == "support":
            support_idx = info["support_idx"]
            support_type = info["support_type"]
            
            if support_type == "pin":
                nodes_df.loc[i, 'rc_x'] = 1
                nodes_df.loc[i, 'rc_y'] = 1
            elif support_type == "roller":
                nodes_df.loc[i, 'rc_y'] = 1
            elif support_type == "fixed":
                nodes_df.loc[i, 'rc_x'] = 1
                nodes_df.loc[i, 'rc_y'] = 1
                nodes_df.loc[i, 'rc_m'] = 1
            elif support_type == "hinge":
                nodes_df.loc[i, 'rc_x'] = 1
                nodes_df.loc[i, 'rc_y'] = 1
    
    # 荷重条件設定（集中荷重・モーメント荷重）
    for l in load_connections:
        # 等分布荷重はスキップ（後で梁に直接設定）
        if l.get("is_udl", False):
            continue
        
        node_idx = l["node_idx"]
        
        if node_idx >= 0 and node_idx < len(nodes_df):
            if l["type"] == "load":
                # 荷重の方向ベクトルを使用
                direction = np.array(l["direction"])
                # FEMライブラリは画像座標系（y下向き正）を使用しているため、そのまま適用
                # 画像: 右=[1,0], 下=[0,1], 左=[-1,0], 上=[0,-1]
                # FEM: 右=ef_x正, 下=ef_y正, 左=ef_x負, 上=ef_y負
                nodes_df.loc[node_idx, 'ef_x'] += direction[0] * load_value
                nodes_df.loc[node_idx, 'ef_y'] += direction[1] * load_value  # そのまま適用
            elif l["type"] == "momentl":
                # momentL = 反時計回り = 正（FEM規則に従う）
                nodes_df.loc[node_idx, 'ef_m'] += -moment_value
            elif l["type"] == "momentr":
                # momentR = 時計回り = 負（FEM規則に従う）
                nodes_df.loc[node_idx, 'ef_m'] += moment_value
    
    # elements_df作成
    elements_df = pd.DataFrame(columns=['young', 'area', 's_moment', 'length', 'angle', 'start', 'end', 'Ws', 'We'])
    
    for conn in beam_connections:
        start_idx = conn["node1_idx"]
        end_idx = conn["node2_idx"]
        
        # 同じ節点同士は接続しない
        if start_idx == end_idx:
            st.warning(f"⚠️ 梁{conn['beam_idx']}: 始点と終点が同じ節点です")
            continue
        
        # 節点座標を取得
        start_coord = np.array(conn["node1_coord"])
        end_coord = np.array(conn["node2_coord"])
        
        # 長さを計算
        length = np.linalg.norm(end_coord - start_coord)
        
        # 長さが極端に短い場合はスキップ
        if length < 1.0:
            st.warning(f"⚠️ 梁{conn['beam_idx']}: 長さが短すぎます ({length:.2f}px)")
            continue
        
        # 角度を再計算（実際の節点座標から）
        angle = math.degrees(math.atan2(end_coord[1] - start_coord[1], 
                                        end_coord[0] - start_coord[0]))
        if angle < 0:
            angle += 360
        
        # 等分布荷重の初期値
        Ws_val = 0
        We_val = 0
        
        elements_df = pd.concat([elements_df, pd.DataFrame([{
            'young': young,
            'area': area,
            's_moment': s_moment,
            'length': length,
            'angle': angle,
            'start': start_idx,
            'end': end_idx,
            'Ws': Ws_val,
            'We': We_val
        }])], ignore_index=True)
    
    # インデックスをリセット
    elements_df = elements_df.reset_index(drop=True)
    
    # 等分布荷重を梁に適用
    for udl in udl_on_beams:
        original_beam_idx = udl["beam_idx"]
        direction = np.array(udl["direction"])
        load_val = udl["load_value"]
        t_start = udl.get("split_t_start", 0)
        t_end = udl.get("split_t_end", 1)
        
        # 分割後の梁の中で、等分布荷重が作用する範囲の梁を特定
        # 元の梁インデックスと一致し、かつt値の範囲内にある梁を探す
        for idx, row in elements_df.iterrows():
            # 元の梁インデックスが一致するか確認
            # beam_connectionsから対応する梁を探す
            if idx < len(beam_connections):
                beam_conn = beam_connections[idx]
                if beam_conn.get("beam_idx") == original_beam_idx:
                    # この梁が等分布荷重の範囲内にあるか確認
                    # 分割された梁の場合、範囲を確認
                    # 簡易的に、元の梁インデックスが一致する全ての梁に適用
                    # （より正確には、t値の範囲を確認する必要がある）
                    
                    # 梁の角度を取得
                    beam_angle = row['angle']
                    beam_angle_rad = np.radians(beam_angle)
                    
                    # 梁の方向ベクトル
                    beam_dir = np.array([np.cos(beam_angle_rad), np.sin(beam_angle_rad)])
                    
                    # 荷重を梁のローカル座標系に変換
                    load_global = np.array([direction[0], direction[1]]) * load_val
                    
                    # 梁の垂直方向成分を計算（梁に垂直な荷重）
                    beam_perp = np.array([-beam_dir[1], beam_dir[0]])
                    load_perp = np.dot(load_global, beam_perp)
                    
                    # 等分布荷重を設定（始点と終点で同じ値）
                    elements_df.loc[idx, 'Ws'] = load_perp
                    elements_df.loc[idx, 'We'] = load_perp

# デバッグ情報（展開可能）
with st.expander("🔍 検出詳細情報"):
    st.write(f"**使用された設定**")
    mode_text = "自動調整" if auto_conf else "手動設定"
    st.write(f"- 検出信頼度: {conf_th:.2f} ({mode_text})")
    st.write(f"- 画像解析サイズ: {img_size}px")
    st.write(f"- IoU閾値: {iou_threshold:.2f}")
    st.write(f"- 最大検出数: {max_det}")
    if enable_preprocessing:
        st.write(f"- 前処理: コントラスト{contrast_factor:.1f}x" + 
                (", 輪郭強調" if edge_enhancement else "") +
                (", ノイズ除去" if noise_reduction else ""))
    st.write(f"- 画像サイズ: {img_width}x{img_height}px")
    
    st.write(f"\n**検出された要素**")
    st.write(f"- 支点: {len(supports)}個")
    st.write(f"- 梁: {len(beams)}個")
    st.write(f"- 荷重: {len(loads)}個")
    st.write(f"- 総節点数: {len(all_nodes)}個")
    
    st.write(f"\n**梁の接続状況**")
    for i, conn in enumerate(beam_connections):
        angle_diff = abs(conn['angle'] - conn.get('original_angle', conn['angle']))
        angle_info = f" [角度補正: {conn.get('original_angle', 0):.1f}° → {conn['angle']:.1f}°]" if angle_diff > 0.1 else ""
        split_info = " [分割済み]" if conn.get('is_split', False) else ""
        st.write(f"梁{i} (元{conn['beam_idx']}): N{conn['node1_idx']} → N{conn['node2_idx']} "
                f"(スナップ距離: {conn['snap1_dist']:.1f}px, {conn['snap2_dist']:.1f}px){angle_info}{split_info}")
    
    st.write(f"\n**荷重の接続状況**")
    for l in load_connections:
        if l.get('is_udl', False):
            # 等分布荷重
            direction = l.get('direction', [0, 0])
            dir_str = f"方向: ({direction[0]:.1f}, {direction[1]:.1f})"
            beam_angle = l.get('closest_beam_angle', 'N/A')
            num_arrows = len(l.get('udl_arrow_positions', []))
            st.write(f"{l['type']}: 矢印数={num_arrows}, 角度: {l['angle']:.0f}°, 梁角度: {beam_angle}°, {dir_str}")
        else:
            # 集中荷重・モーメント荷重
            split_info = " [梁を分割]" if l.get('needs_split', False) else ""
            direction = l.get('direction', [0, 0])
            dir_str = f", 方向: ({direction[0]:.1f}, {direction[1]:.1f})"
            st.write(f"{l['type']}: 節点N{l['node_idx']} (梁{l['on_beam']}, t={l['beam_t']:.2f}, 距離: {l['dist_to_beam']:.1f}px, 角度: {l['angle']:.0f}°{dir_str}){split_info}")
    
    if udl_on_beams:
        st.write(f"\n**等分布荷重が作用する梁**")
        for udl in udl_on_beams:
            beam_idx = udl["beam_idx"]
            load_val = udl["load_value"]
            direction = udl["direction"]
            t_start = udl.get("t_start", 0)
            t_end = udl.get("t_end", 1)
            st.write(f"梁{beam_idx}: 荷重値={load_val:.1f}, 方向=({direction[0]:.1f}, {direction[1]:.1f}), 範囲=t[{t_start:.2f}, {t_end:.2f}]")
    
    st.write(f"\n**節点一覧**")
    for i, (node, info) in enumerate(zip(all_nodes, node_info)):
        node_coord = node if isinstance(node, np.ndarray) else np.array(node)
        st.write(f"N{i}: ({node_coord[0]:.1f}, {node_coord[1]:.1f}) - {info['type']}")

st.subheader("📋 解析データ")

# 構造の妥当性チェック
if len(elements_df) == 0:
    st.error("❌ 部材が検出されませんでした。梁が正しく認識されているか確認してください。")
    st.stop()

if len(nodes_df) == 0:
    st.error("❌ 節点が検出されませんでした。")
    st.stop()

# 拘束条件のチェック
constraint_count = nodes_df[['rc_x', 'rc_y', 'rc_m']].sum().sum()
if constraint_count < 3:
    st.warning("⚠️ 拘束条件が不足している可能性があります（最低3つの拘束が必要）")

tab1, tab2, tab3 = st.tabs(["節点情報", "部材情報", "荷重・拘束条件"])

with tab1:
    st.write(f"**節点数: {len(nodes_df)}**")
    display_nodes = nodes_df.copy()
    display_nodes.index.name = '節点番号'
    st.dataframe(display_nodes[['x', 'y']], use_container_width=True)

with tab2:
    st.write(f"**部材数: {len(elements_df)}**")
    st.dataframe(elements_df[['start', 'end', 'length', 'angle', 'young', 'area', 's_moment']], use_container_width=True)

with tab3:
    constraint_df = nodes_df[nodes_df[['rc_x', 'rc_y', 'rc_m']].sum(axis=1) > 0]
    load_df = nodes_df[nodes_df[['ef_x', 'ef_y', 'ef_m']].abs().sum(axis=1) > 0]
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**拘束条件 ({len(constraint_df)}節点)**")
        if len(constraint_df) > 0:
            st.dataframe(constraint_df[['x', 'y', 'rc_x', 'rc_y', 'rc_m']], use_container_width=True)
        else:
            st.warning("拘束条件が設定されていません")
    with col_b:
        st.write(f"**荷重条件 ({len(load_df)}節点)**")
        if len(load_df) > 0:
            st.dataframe(load_df[['x', 'y', 'ef_x', 'ef_y', 'ef_m']], use_container_width=True)
        else:
            st.info("荷重が設定されていません")

def adjust_stress_data_to_corrected_beams(fig_list, beam_connections):
    """応力図データを15度補正済みの部材に合わせて調整"""
    adjusted_fig_list = []
    
    for i, df in enumerate(fig_list):
        if i >= len(beam_connections):
            continue
            
        df_adjusted = df.copy()
        conn = beam_connections[i]
        
        # 元の部材の端点
        pt1_orig = np.array(conn["node1_coord"])
        pt2_orig = np.array(conn["node2_coord"])
        
        # 15度刻みに補正された部材の端点
        vector = pt2_orig - pt1_orig
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        corrected_angle = round(angle / 15) * 15
        length = np.linalg.norm(vector)
        angle_rad = math.radians(corrected_angle)
        pt2_corrected = pt1_orig + length * np.array([math.cos(angle_rad), math.sin(angle_rad)])
        
        # 部材上の各点を補正済み部材上の対応点に変換
        for j in range(len(df_adjusted)):
            # 元の部材上での位置比率を計算
            orig_point = np.array([df.iloc[j]['x'], df.iloc[j]['y']])
            
            # 元の部材の方向ベクトル
            orig_vector = pt2_orig - pt1_orig
            orig_length = np.linalg.norm(orig_vector)
            
            if orig_length > 0:
                # 元の部材上での位置比率
                t = np.dot(orig_point - pt1_orig, orig_vector) / (orig_length ** 2)
                t = max(0, min(1, t))  # 0-1の範囲にクランプ
                
                # 補正済み部材上の対応点
                corrected_point = pt1_orig + t * (pt2_corrected - pt1_orig)
                
                # 座標を更新
                df_adjusted.iloc[j, df_adjusted.columns.get_loc('x')] = corrected_point[0]
                df_adjusted.iloc[j, df_adjusted.columns.get_loc('y')] = corrected_point[1]
                
                # 応力図の座標も同様に調整
                if 'Nx' in df_adjusted.columns:
                    # 軸力図の座標調整
                    stress_offset = np.array([df.iloc[j]['Nx'] - df.iloc[j]['x'], df.iloc[j]['Ny'] - df.iloc[j]['y']])
                    df_adjusted.iloc[j, df_adjusted.columns.get_loc('Nx')] = corrected_point[0] + stress_offset[0]
                    df_adjusted.iloc[j, df_adjusted.columns.get_loc('Ny')] = corrected_point[1] + stress_offset[1]
                
                if 'Qx' in df_adjusted.columns:
                    # せん断力図の座標調整
                    stress_offset = np.array([df.iloc[j]['Qx'] - df.iloc[j]['x'], df.iloc[j]['Qy'] - df.iloc[j]['y']])
                    df_adjusted.iloc[j, df_adjusted.columns.get_loc('Qx')] = corrected_point[0] + stress_offset[0]
                    df_adjusted.iloc[j, df_adjusted.columns.get_loc('Qy')] = corrected_point[1] + stress_offset[1]
                
                if 'Mx' in df_adjusted.columns:
                    # 曲げモーメント図の座標調整
                    stress_offset = np.array([df.iloc[j]['Mx'] - df.iloc[j]['x'], df.iloc[j]['My'] - df.iloc[j]['y']])
                    df_adjusted.iloc[j, df_adjusted.columns.get_loc('Mx')] = corrected_point[0] + stress_offset[0]
                    df_adjusted.iloc[j, df_adjusted.columns.get_loc('My')] = corrected_point[1] + stress_offset[1]
                
                # 変形図の座標も調整
                if 'ax' in df_adjusted.columns:
                    deform_offset = np.array([df.iloc[j]['ax'] - df.iloc[j]['x'], df.iloc[j]['ay'] - df.iloc[j]['y']])
                    df_adjusted.iloc[j, df_adjusted.columns.get_loc('ax')] = corrected_point[0] + deform_offset[0]
                    df_adjusted.iloc[j, df_adjusted.columns.get_loc('ay')] = corrected_point[1] + deform_offset[1]
        
        adjusted_fig_list.append(df_adjusted)
    
    return adjusted_fig_list

# FEM解析実行
try:
    with st.spinner("FEM解析実行中..."):
        D_R, M_S = fem_lib.fem_calc(elements_df, nodes_df)
    
    st.success("✅ FEM解析完了")
    
    # 結果表示
    st.subheader("📊 解析結果")
    
    # 変位・反力の表示
    st.write("**節点変位・反力**")
    st.dataframe(D_R, use_container_width=True)
    
    # 変形図の表示
    st.write("**変形図**")
    
    # draw_lib.make_figureを使用して変形図を作成
    fig_list_deform = draw_lib.make_figure(M_S)
    
    # 応力図データを15度補正済み部材に合わせて調整
    fig_list_deform_adjusted = adjust_stress_data_to_corrected_beams(fig_list_deform, beam_connections)
    
    # 変形量のスケールを拡大（最大変位を構造の1/10程度に）
    max_displacement = 0
    for df in fig_list_deform_adjusted:
        for i in range(len(df)):
            dx = df.loc[i, 'ax'] - df.loc[i, 'x']
            dy = df.loc[i, 'ay'] - df.loc[i, 'y']
            disp = np.sqrt(dx**2 + dy**2)
            max_displacement = max(max_displacement, disp)
    
    # 構造の代表長さを計算
    all_coords = []
    for conn in beam_connections:
        all_coords.append(conn["node1_coord"])
        all_coords.append(conn["node2_coord"])
    all_coords = np.array(all_coords)
    structure_size = np.max(np.ptp(all_coords, axis=0))
    
    # スケール係数を計算（構造の1/10を目標）
    if max_displacement > 1e-6:
        scale_factor = (structure_size / 10) / max_displacement
    else:
        scale_factor = 1.0
    
    # 変形をスケール拡大
    fig_list_deform_scaled = []
    for df in fig_list_deform_adjusted:
        df_scaled = df.copy()
        for i in range(len(df_scaled)):
            dx = df_scaled.loc[i, 'ax'] - df_scaled.loc[i, 'x']
            dy = df_scaled.loc[i, 'ay'] - df_scaled.loc[i, 'y']
            df_scaled.loc[i, 'ax'] = df_scaled.loc[i, 'x'] + dx * scale_factor
            df_scaled.loc[i, 'ay'] = df_scaled.loc[i, 'y'] + dy * scale_factor
        fig_list_deform_scaled.append(df_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    
    # 変形前の形状（グレー）- 15度刻みで角度補正
    for conn in beam_connections:
        pt1 = np.array(conn["node1_coord"])
        pt2 = np.array(conn["node2_coord"])
        
        # 15度刻みに角度を補正
        vector = pt2 - pt1
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        corrected_angle = round(angle / 15) * 15
        
        # 補正後の座標を計算
        length = np.linalg.norm(vector)
        angle_rad = math.radians(corrected_angle)
        pt2_corrected = pt1 + length * np.array([math.cos(angle_rad), math.sin(angle_rad)])
        
        ax.plot([pt1[0], pt2_corrected[0]], [pt1[1], pt2_corrected[1]], 'gray', linewidth=6, alpha=0.7)
    
    # 変形後の形状（黒）- 調整済みデータを使用
    for df in fig_list_deform_scaled:
        ax.plot(df['ax'], df['ay'], 'black', linewidth=6)
    
    # 節点（15度刻みで調整された位置に表示）
    adjusted_node_positions = {}
    
    # 各梁接続から15度補正済みの節点位置を計算
    for conn in beam_connections:
        node1_idx = conn["node1_idx"]
        node2_idx = conn["node2_idx"]
        
        pt1_orig = np.array(conn["node1_coord"])
        pt2_orig = np.array(conn["node2_coord"])
        
        # 15度刻みに角度を補正
        vector = pt2_orig - pt1_orig
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        corrected_angle = round(angle / 15) * 15
        
        # 補正後の座標を計算
        length = np.linalg.norm(vector)
        angle_rad = math.radians(corrected_angle)
        pt2_corrected = pt1_orig + length * np.array([math.cos(angle_rad), math.sin(angle_rad)])
        
        # 調整済み節点位置を記録
        adjusted_node_positions[node1_idx] = pt1_orig
        adjusted_node_positions[node2_idx] = pt2_corrected
    
    # 調整済み節点位置を表示
    for node_idx, pos in adjusted_node_positions.items():
        ax.plot(pos[0], pos[1], 'ko', markersize=8)
        ax.text(pos[0], pos[1], f'  N{node_idx}', fontsize=10)
    
    # 軸、タイトル、枠線を削除
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.invert_yaxis()
    st.pyplot(fig)
    
    # 応力図の表示
    st.write("**応力図**")
    
    # 応力図用のデータを作成（スケール調整なし）
    fig_list_original = draw_lib.make_figure(M_S)
    
    # 応力図データを15度補正済み部材に合わせて調整
    fig_list_original_adjusted = adjust_stress_data_to_corrected_beams(fig_list_original, beam_connections)
    
    # 平均部材長を計算
    avg_beam_length = elements_df['length'].mean() if len(elements_df) > 0 else 100
    target_stress_display = avg_beam_length / 4  # 最大応力を部材長の1/4に
    
    # 各応力の最大値を計算
    max_N = max([abs(df['N']).max() for df in fig_list_original_adjusted] + [1e-6])
    max_Q = max([abs(df['Q']).max() for df in fig_list_original_adjusted] + [1e-6])
    max_M = max([abs(df['M']).max() for df in fig_list_original_adjusted] + [1e-6])
    
    # スケール係数を計算
    scale_N = target_stress_display / max_N
    scale_Q = target_stress_display / max_Q
    scale_M = target_stress_display / max_M
    
    # スケール調整した応力図データを作成
    fig_list = []
    for df in fig_list_original_adjusted:
        df_scaled = df.copy()
        # 応力値をスケール調整
        df_scaled['N'] = df['N'] * scale_N
        df_scaled['Q'] = df['Q'] * scale_Q
        df_scaled['M'] = df['M'] * scale_M
        # 座標もスケール調整
        df_scaled['Nx'] = df['x'] + (df['Nx'] - df['x']) * scale_N
        df_scaled['Ny'] = df['y'] + (df['Ny'] - df['y']) * scale_N
        df_scaled['Qx'] = df['x'] + (df['Qx'] - df['x']) * scale_Q
        df_scaled['Qy'] = df['y'] + (df['Qy'] - df['y']) * scale_Q
        df_scaled['Mx'] = df['x'] + (df['Mx'] - df['x']) * scale_M
        df_scaled['My'] = df['y'] + (df['My'] - df['y']) * scale_M
        fig_list.append(df_scaled)
    
    # 軸力図(N)
    st.write("**軸力図 (N)**")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    
    # 部材を15度刻みで表示
    for conn in beam_connections:
        pt1 = np.array(conn["node1_coord"])
        pt2 = np.array(conn["node2_coord"])
        
        # 15度刻みに角度を補正
        vector = pt2 - pt1
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        corrected_angle = round(angle / 15) * 15
        
        # 補正後の座標を計算
        length = np.linalg.norm(vector)
        angle_rad = math.radians(corrected_angle)
        pt2_corrected = pt1 + length * np.array([math.cos(angle_rad), math.sin(angle_rad)])
        
        ax.plot([pt1[0], pt2_corrected[0]], [pt1[1], pt2_corrected[1]], 'black', linewidth=6)
    
    for df in fig_list:
        ax.plot(df['Nx'], df['Ny'], 'b-', linewidth=3)
        ax.fill(list(df['x']) + list(df['Nx'][::-1]), 
               list(df['y']) + list(df['Ny'][::-1]), 
               'blue', alpha=0.3)
    
    # 軸、タイトル、枠線を削除
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.invert_yaxis()
    st.pyplot(fig)
    
    # せん断力図(Q)
    st.write("**せん断力図 (Q)**")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    
    # 部材を15度刻みで表示
    for conn in beam_connections:
        pt1 = np.array(conn["node1_coord"])
        pt2 = np.array(conn["node2_coord"])
        
        # 15度刻みに角度を補正
        vector = pt2 - pt1
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        corrected_angle = round(angle / 15) * 15
        
        # 補正後の座標を計算
        length = np.linalg.norm(vector)
        angle_rad = math.radians(corrected_angle)
        pt2_corrected = pt1 + length * np.array([math.cos(angle_rad), math.sin(angle_rad)])
        
        ax.plot([pt1[0], pt2_corrected[0]], [pt1[1], pt2_corrected[1]], 'black', linewidth=6)
    
    for df in fig_list:
        ax.plot(df['Qx'], df['Qy'], 'g-', linewidth=3)
        ax.fill(list(df['x']) + list(df['Qx'][::-1]), 
               list(df['y']) + list(df['Qy'][::-1]), 
               'green', alpha=0.3)
    
    # 軸、タイトル、枠線を削除
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.invert_yaxis()
    st.pyplot(fig)
    
    # 曲げモーメント図(M)
    st.write("**曲げモーメント図 (M)**")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    
    # 部材を15度刻みで表示
    for conn in beam_connections:
        pt1 = np.array(conn["node1_coord"])
        pt2 = np.array(conn["node2_coord"])
        
        # 15度刻みに角度を補正
        vector = pt2 - pt1
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        corrected_angle = round(angle / 15) * 15
        
        # 補正後の座標を計算
        length = np.linalg.norm(vector)
        angle_rad = math.radians(corrected_angle)
        pt2_corrected = pt1 + length * np.array([math.cos(angle_rad), math.sin(angle_rad)])
        
        ax.plot([pt1[0], pt2_corrected[0]], [pt1[1], pt2_corrected[1]], 'black', linewidth=6)
    
    for df in fig_list:
        ax.plot(df['Mx'], df['My'], 'r-', linewidth=3)
        ax.fill(list(df['x']) + list(df['Mx'][::-1]), 
               list(df['y']) + list(df['My'][::-1]), 
               'red', alpha=0.3)
    
    # 軸、タイトル、枠線を削除
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.invert_yaxis()
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ 解析エラー: {str(e)}")
    st.exception(e)
