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
    
    # 回転行列を適用（画像座標系: y下向き正、時計回りが正）
    # テンプレートの基準は下向き（90度）なので、-90度オフセットを適用
    theta = np.deg2rad(angle - 90)
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
    conf_th = st.slider("検出信頼度", 0.2, 1.0, 0.45, 0.01)
    
    # 固定値設定
    y_align_th = 150.0
    node_connect_th = 250
    young = 2.0e2
    area = 9.0e2
    s_moment = 6.75e4
    load_value = 10.0
    moment_value = 10.0
    udl_value = 5.0
    
    st.markdown("---")
    st.subheader("📋 固定設定値")
    st.markdown(f"""
    **閾値設定**
    - 高さ揃え閾値: `{y_align_th:.0f}px`
    - 接続閾値: `{node_connect_th}px`
    
    **材料特性**
    - ヤング係数 E: `{young:.1e}`
    - 断面積 A: `{area:.1e}`
    - 断面二次モーメント I: `{s_moment:.1e}`
    
    **荷重設定**
    - 集中荷重: `{load_value:.1f}`
    - モーメント荷重: `{moment_value:.1f}`
    - 等分布荷重: `{udl_value:.1f}`
    """)

uploaded = st.file_uploader("📷 構造図画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("画像ファイルをアップロードしてください")
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

col1, col2 = st.columns(2)
with col1:
    st.image(img_pil, caption="元画像", use_container_width=True)

TEMPL = {k: load_template_rgba(template_path(k)) for k in TEMPLATE_FILES}

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    st.error(f"モデルパスが存在しません: {MODEL_PATH}")
    st.stop()

if not st.button("🚀 解析実行", type="primary"):
    st.stop()

with st.spinner("画像認識中..."):
    model = YOLO(MODEL_PATH)
    res = model(img, conf=conf_th, imgsz=640)[0]
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
    angle = round_angle_deg(
        math.degrees(math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])) if name != "beam" else
        math.degrees(math.atan2(pts[2][1] - pts[0][1], pts[2][0] - pts[0][0]))
    )
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
        loads.append({"type": name, "pts": pts, "angle": round_angle_deg(angle), "conf": conf})

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

# 2. 梁端点を既存の節点にスナップ、または新規節点として追加
beam_connections = []
for be in beam_endpoints:
    # 端点1の処理
    pt1 = be["pt1"]
    min_dist1 = float('inf')
    snap_idx1 = -1
    
    for i, node in enumerate(all_nodes):
        dist = np.linalg.norm(pt1 - node)
        if dist < min_dist1:
            min_dist1 = dist
            snap_idx1 = i
    
    # 閾値内ならスナップ、そうでなければ新規節点
    if min_dist1 < node_connect_th and snap_idx1 >= 0:
        node1_idx = snap_idx1
        node1_coord = all_nodes[snap_idx1]
    else:
        # 新規節点として追加
        node1_idx = len(all_nodes)
        node1_coord = pt1
        all_nodes.append(pt1)
        node_info.append({"type": "beam_endpoint", "beam_idx": be["beam_idx"]})
    
    # 端点2の処理
    pt2 = be["pt2"]
    min_dist2 = float('inf')
    snap_idx2 = -1
    
    for i, node in enumerate(all_nodes):
        dist = np.linalg.norm(pt2 - node)
        if dist < min_dist2:
            min_dist2 = dist
            snap_idx2 = i
    
    # 閾値内ならスナップ、そうでなければ新規節点
    if min_dist2 < node_connect_th and snap_idx2 >= 0:
        node2_idx = snap_idx2
        node2_coord = all_nodes[snap_idx2]
    else:
        # 新規節点として追加
        node2_idx = len(all_nodes)
        node2_coord = pt2
        all_nodes.append(pt2)
        node_info.append({"type": "beam_endpoint", "beam_idx": be["beam_idx"]})
    
    # ===== 梁の角度を15度刻みに補正 + 垂直接続の検出 =====
    # 現在の角度を計算
    node1_arr = np.array(node1_coord) if not isinstance(node1_coord, np.ndarray) else node1_coord
    node2_arr = np.array(node2_coord) if not isinstance(node2_coord, np.ndarray) else node2_coord
    
    current_angle = math.degrees(math.atan2(node2_arr[1] - node1_arr[1], 
                                            node2_arr[0] - node1_arr[0]))
    if current_angle < 0:
        current_angle += 360
    
    # 15度刻みに丸める
    corrected_angle = round(current_angle / 15) * 15
    
    # 垂直接続の検出: 他の梁と90度に近い場合は90度にする
    for existing_conn in beam_connections:
        existing_angle = existing_conn.get("angle", 0)
        angle_diff = abs(corrected_angle - existing_angle)
        
        # 角度差を0-180度の範囲に正規化
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 90度に近い場合（85-95度の範囲）
        if 85 <= angle_diff <= 95:
            # 正確に90度にする
            if angle_diff < 90:
                corrected_angle = existing_angle + 90
            else:
                corrected_angle = existing_angle - 90
            
            # 0-360度の範囲に正規化
            corrected_angle = corrected_angle % 360
            break
    
    # 角度が変わった場合、端点2の座標を補正
    if abs(corrected_angle - current_angle) > 0.1:
        # 梁の長さを保持
        beam_length = np.linalg.norm(node2_arr - node1_arr)
        
        # 補正後の角度で端点2の新しい座標を計算
        angle_rad = math.radians(corrected_angle)
        new_node2_x = node1_arr[0] + beam_length * math.cos(angle_rad)
        new_node2_y = node1_arr[1] + beam_length * math.sin(angle_rad)
        node2_coord_corrected = np.array([new_node2_x, new_node2_y])
        
        # 端点2が新規節点の場合のみ座標を更新
        if node2_idx == len(all_nodes) - 1 and node_info[-1]["type"] == "beam_endpoint":
            all_nodes[node2_idx] = node2_coord_corrected
            node2_coord = node2_coord_corrected
        
        # 端点1が新規節点の場合も同様に補正（端点1を固定して端点2を動かす方が自然）
        # ただし、両端点が支点にスナップしている場合は補正しない
    else:
        corrected_angle = current_angle
    
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
        # UDLボックスのx座標範囲を取得
        pts = l["pts"]
        x_min = np.min(pts[:, 0])
        x_max = np.max(pts[:, 0])
        y_min = np.min(pts[:, 1])
        y_max = np.max(pts[:, 1])
        box_center = pts.mean(axis=0)
        
        # 荷重の向きを角度から判定
        # 0度=右、90度=下、180度=左、270度=上
        if 45 <= angle < 135:  # 下向き
            load_direction = np.array([0, -1])
        elif 135 <= angle < 225:  # 左向き
            load_direction = np.array([-1, 0])
        elif 225 <= angle < 315:  # 上向き
            load_direction = np.array([0, 1])
        else:  # 右向き
            load_direction = np.array([1, 0])
        
        # この範囲に重なる梁を探す
        for idx, beam in enumerate(beam_connections):
            a = np.array(beam["node1_coord"])
            b = np.array(beam["node2_coord"])
            
            # 梁のx座標範囲
            beam_x_min = min(a[0], b[0])
            beam_x_max = max(a[0], b[0])
            
            # x座標範囲が重なるかチェック
            overlap_x_min = max(x_min, beam_x_min)
            overlap_x_max = min(x_max, beam_x_max)
            
            if overlap_x_min < overlap_x_max:
                # 梁の中心がUDLボックスの近くにあるかチェック
                beam_center = (a + b) / 2
                dist_to_box = np.linalg.norm(beam_center - box_center)
                
                # 距離が閾値以内なら等分布荷重を適用
                if dist_to_box < 150:  # 閾値
                    udl_on_beams.append({
                        "beam_idx": idx,
                        "load_value": udl_value,
                        "direction": load_direction,
                        "angle": angle,
                        "x_range": (overlap_x_min, overlap_x_max),
                        "box_range": (x_min, x_max, y_min, y_max)
                    })
        
        # 等分布荷重の接続情報を記録（表示用）
        # 最も近い梁の中点に表示位置を設定
        display_coord = box_center
        if udl_on_beams:
            # 最初に見つかった梁の中点を使用
            first_beam_idx = udl_on_beams[-1]["beam_idx"]
            if first_beam_idx < len(beam_connections):
                beam = beam_connections[first_beam_idx]
                a = np.array(beam["node1_coord"])
                b = np.array(beam["node2_coord"])
                display_coord = (a + b) / 2  # 梁の中点
        
        load_connections.append({
            "type": load_type,
            "tip_coord": box_center.tolist(),
            "proj_coord": display_coord.tolist() if isinstance(display_coord, np.ndarray) else display_coord,
            "node_idx": -1,
            "on_beam": udl_on_beams[-1]["beam_idx"] if udl_on_beams else -1,
            "beam_idx_in_list": -1,
            "beam_t": 0.5,
            "angle": angle,
            "conf": float(l["conf"]),
            "dist_to_beam": 0,
            "needs_split": False,
            "is_udl": True,
            "x_range": (x_min, x_max),
            "direction": load_direction.tolist()
        })
        continue
    
    # 集中荷重・モーメント荷重の処理
    # テンプレートの中心座標
    center = l["pts"].mean(axis=0)
    
    if load_type in ["load"]:
        # テンプレートを使って回転後の矢じり先端座標を計算
        tpl = TEMPL.get(load_type)
        if tpl is not None:
            tip = get_rotated_arrow_tip(tpl, center, angle)
        else:
            # フォールバック: ボックスの中心
            tip = center
    else:  # moment
        tip = center
    
    # 荷重の向きを角度から判定
    if 45 <= angle < 135:  # 下向き
        load_direction = np.array([0, -1])
    elif 135 <= angle < 225:  # 左向き
        load_direction = np.array([-1, 0])
    elif 225 <= angle < 315:  # 上向き
        load_direction = np.array([0, 1])
    else:  # 右向き
        load_direction = np.array([1, 0])
    
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
        "direction": load_direction.tolist()
    })

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

# ===== 清書画像生成 =====
cleaned = np.ones_like(img) * 255

# 梁を描画（接続後の節点座標を使用）
for conn in beam_connections:
    # 接続後の節点座標を取得
    node1_idx = conn["node1_idx"]
    node2_idx = conn["node2_idx"]
    pt1 = np.array(all_nodes[node1_idx])
    pt2 = np.array(all_nodes[node2_idx])
    cv2.line(cleaned, tuple(map(int, pt1)), tuple(map(int, pt2)), (80, 80, 80), 4)

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
    
    # 等分布荷重でない場合、矢じり先端と梁上の点を線で接続
    if not l.get("is_udl", False):
        # 接続線を描画（緑色の細線）
        cv2.line(cleaned, tuple(map(int, tip)), tuple(map(int, proj)), (0, 200, 0), 2)
        
        # 矢じり先端に小さな円を描画（荷重の作用点）
        cv2.circle(cleaned, tuple(map(int, tip)), 5, (0, 0, 255), -1)
        
        # 梁上の接続点に円を描画
        cv2.circle(cleaned, tuple(map(int, proj)), 6, (255, 0, 0), 2)
    
    # 荷重テンプレートを配置（矢じりが梁上の接続点 proj に来るように）
    if tpl is not None:
        tpl_scaled = scale_image(tpl, 0.9)
        # テンプレートの基準は下向き（90度）なので、-90度オフセットを適用
        tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle - 90)
        
        # 回転後のテンプレート内の矢じり位置を取得
        h_rot, w_rot = tpl_rot.shape[:2]
        tip_local_rot = get_template_arrow_tip(tpl_rot)
        
        # テンプレート中心からのオフセット (row, col)
        offset_row = tip_local_rot[0] - h_rot // 2
        offset_col = tip_local_rot[1] - w_rot // 2
        
        # (x, y) 座標系に変換
        offset_x = offset_col
        offset_y = offset_row
        
        # テンプレート中心位置を計算（矢じりが梁上の接続点 proj に来るように）
        template_center = proj - np.array([offset_x, offset_y])
        
        cleaned = overlay_rgba(cleaned, tpl_rot, template_center)

with col2:
    st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), "清書画像", use_container_width=True)

st.success("✅ 画像認識・清書完了")

# ===== FEM解析用データ構造への変換 =====
with st.spinner("FEM解析データ準備中..."):
    # all_nodesをそのまま使用（既に重複排除済み）
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
                # 荷重の方向ベクトルを使用（FEM規則: x右向き正、y上向き正）
                direction = np.array(l["direction"])
                # 画像座標系（y下向き正）からFEM座標系（y上向き正）に変換
                nodes_df.loc[node_idx, 'ef_x'] += direction[0] * load_value
                nodes_df.loc[node_idx, 'ef_y'] += -direction[1] * load_value  # y軸反転
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
        beam_idx = udl["beam_idx"]
        if beam_idx < len(elements_df):
            direction = np.array(udl["direction"])
            load_val = udl["load_value"]
            
            # 梁の角度を取得
            beam_angle = elements_df.loc[beam_idx, 'angle']
            beam_angle_rad = np.radians(beam_angle)
            
            # 梁の方向ベクトル
            beam_dir = np.array([np.cos(beam_angle_rad), np.sin(beam_angle_rad)])
            
            # 荷重を梁のローカル座標系に変換
            # 画像座標系（y下向き正）からFEM座標系（y上向き正）に変換
            load_global = np.array([direction[0], -direction[1]]) * load_val
            
            # 梁の垂直方向成分を計算（梁に垂直な荷重）
            beam_perp = np.array([-beam_dir[1], beam_dir[0]])
            load_perp = np.dot(load_global, beam_perp)
            
            # 等分布荷重を設定（始点と終点で同じ値）
            elements_df.loc[beam_idx, 'Ws'] = load_perp
            elements_df.loc[beam_idx, 'We'] = load_perp

# デバッグ情報（展開可能）
with st.expander("🔍 検出詳細情報"):
    st.write(f"**検出された要素**")
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
            x_range = l.get('x_range', (0, 0))
            direction = l.get('direction', [0, 0])
            dir_str = f"方向: ({direction[0]:.1f}, {direction[1]:.1f})"
            st.write(f"{l['type']}: x範囲 [{x_range[0]:.1f}, {x_range[1]:.1f}], 角度: {l['angle']:.0f}°, {dir_str}")
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
            x_range = udl["x_range"]
            st.write(f"梁{beam_idx}: 荷重値={load_val:.1f}, 方向=({direction[0]:.1f}, {direction[1]:.1f}), x範囲=[{x_range[0]:.1f}, {x_range[1]:.1f}]")
    
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

# FEM解析実行
try:
    with st.spinner("FEM解析実行中..."):
        D_R, M_S = fem_lib.fem_calc(elements_df, nodes_df)
    
    st.success("✅ FEM解析完了")
    
    # 結果表示
    st.subheader("📊 解析結果")
    
    tab_r1, tab_r2, tab_r3 = st.tabs(["変位・反力", "変形図", "応力図"])
    
    with tab_r1:
        st.write("**節点変位・反力**")
        st.dataframe(D_R, use_container_width=True)
    
    with tab_r2:
        # draw_lib.make_figureを使用して変形図を作成
        fig_list_deform = draw_lib.make_figure(M_S)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title("変形図", fontsize=16, fontweight='bold')
        
        # 元の形状（灰色）
        for conn in beam_connections:
            pt1 = np.array(conn["node1_coord"])
            pt2 = np.array(conn["node2_coord"])
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'gray', linewidth=2, alpha=0.3, label='元形状' if conn == beam_connections[0] else '')
        
        # 変形後の形状（赤色）
        for df in fig_list_deform:
            ax.plot(df['ax'], df['ay'], 'r-', linewidth=2, label='変形後' if df is fig_list_deform[0] else '')
        
        # 節点
        for i, row in nodes_df.iterrows():
            ax.plot(row['x'], row['y'], 'ko', markersize=8)
            ax.text(row['x'], row['y'], f'  N{i}', fontsize=10)
        
        ax.legend()
        ax.invert_yaxis()
        st.pyplot(fig)
    
    with tab_r3:
        # 応力図用のデータを作成（スケール調整なし）
        fig_list_original = draw_lib.make_figure(M_S)
        
        # 平均部材長を計算
        avg_beam_length = elements_df['length'].mean() if len(elements_df) > 0 else 100
        target_stress_display = avg_beam_length / 4  # 最大応力を部材長の1/4に
        
        # 各応力の最大値を計算
        max_N = max([abs(df['N']).max() for df in fig_list_original] + [1e-6])
        max_Q = max([abs(df['Q']).max() for df in fig_list_original] + [1e-6])
        max_M = max([abs(df['M']).max() for df in fig_list_original] + [1e-6])
        
        # スケール係数を計算
        scale_N = target_stress_display / max_N
        scale_Q = target_stress_display / max_Q
        scale_M = target_stress_display / max_M
        
        # スケール調整した応力図データを作成
        fig_list = []
        for df in fig_list_original:
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
        
        stress_tabs = st.tabs(["軸力図(N)", "せん断力図(Q)", "曲げモーメント図(M)"])
        
        with stress_tabs[0]:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title("軸力図 (N)", fontsize=16, fontweight='bold')
            
            for conn in beam_connections:
                pt1 = np.array(conn["node1_coord"])
                pt2 = np.array(conn["node2_coord"])
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'gray', linewidth=2, alpha=0.3)
            
            for df in fig_list:
                ax.plot(df['x'], df['y'], 'k-', linewidth=1)
                ax.plot(df['Nx'], df['Ny'], 'b-', linewidth=2)
                ax.fill(list(df['x']) + list(df['Nx'][::-1]), 
                       list(df['y']) + list(df['Ny'][::-1]), 
                       'blue', alpha=0.3)
            
            ax.invert_yaxis()
            st.pyplot(fig)
        
        with stress_tabs[1]:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title("せん断力図 (Q)", fontsize=16, fontweight='bold')
            
            for conn in beam_connections:
                pt1 = np.array(conn["node1_coord"])
                pt2 = np.array(conn["node2_coord"])
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'gray', linewidth=2, alpha=0.3)
            
            for df in fig_list:
                ax.plot(df['x'], df['y'], 'k-', linewidth=1)
                ax.plot(df['Qx'], df['Qy'], 'g-', linewidth=2)
                ax.fill(list(df['x']) + list(df['Qx'][::-1]), 
                       list(df['y']) + list(df['Qy'][::-1]), 
                       'green', alpha=0.3)
            
            ax.invert_yaxis()
            st.pyplot(fig)
        
        with stress_tabs[2]:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title("曲げモーメント図 (M)", fontsize=16, fontweight='bold')
            
            for conn in beam_connections:
                pt1 = np.array(conn["node1_coord"])
                pt2 = np.array(conn["node2_coord"])
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'gray', linewidth=2, alpha=0.3)
            
            for df in fig_list:
                ax.plot(df['x'], df['y'], 'k-', linewidth=1)
                ax.plot(df['Mx'], df['My'], 'r-', linewidth=2)
                ax.fill(list(df['x']) + list(df['Mx'][::-1]), 
                       list(df['y']) + list(df['My'][::-1]), 
                       'red', alpha=0.3)
            
            ax.invert_yaxis()
            st.pyplot(fig)
    
    st.balloons()

except Exception as e:
    st.error(f"❌ 解析エラー: {str(e)}")
    st.exception(e)
