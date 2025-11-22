# seisyo_final_fixed_v5_nodes.py
import os
import math
import json
import io
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from typing import Any, Tuple, List, Dict

st.set_page_config(layout="wide", page_title="構造図 清書 + 可視化 (fixed-v5 - nodes)")

# ---------------------------
# 設定
# ---------------------------
DEFAULT_MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
TEMPLATE_DIR = "templates"

# ---------------------------
# ユーティリティ: 安全に Tensor -> numpy
# ---------------------------
def to_numpy(x: Any):
    try:
        return x.cpu().numpy()
    except Exception:
        try:
            return np.array(x)
        except Exception:
            return x

# ---------------------------
# 頂点の順序安定化（CW、開始点は top-left に）
# ---------------------------
def order_pts_clockwise_start(pts: np.ndarray, start_hint: str = "top-left") -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1,2)
    if pts.shape[0] != 4:
        return pts
    cx = pts[:,0].mean()
    cy = pts[:,1].mean()
    angles = np.arctan2(pts[:,1] - cy, pts[:,0] - cx)
    order = np.argsort(-angles)  # CW
    pts_sorted = pts[order]
    # choose start index
    if start_hint == "top-left":
        ys = pts_sorted[:,1]
        miny = ys.min()
        cand = np.where(np.isclose(ys, miny, atol=1e-6))[0]
        if len(cand) > 1:
            xs = pts_sorted[cand,0]
            idx_start = int(cand[np.argmin(xs)])
        else:
            idx_start = int(cand[0])
    else:
        idx_start = 0
    pts_final = np.roll(pts_sorted, -idx_start, axis=0)
    return pts_final

# ---------------------------
# RGBA テンプレ読み込み
# ---------------------------
def load_template_rgba(path: str):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        b,g,r = cv2.split(img)
        a = np.ones_like(b) * 255
        img = cv2.merge([b,g,r,a])
    return img

# ---------------------------
# 回転・スケール・合成
# ---------------------------
def rotate_image_keep_alpha(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h,w = img.shape[:2]
    cx, cy = w/2.0, h/2.0
    M = cv2.getRotationMatrix2D((cx,cy), angle_deg, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0,2] += (nw/2.0 - cx)
    M[1,2] += (nh/2.0 - cy)
    rotated = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def scale_image(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return img
    h,w = img.shape[:2]
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def overlay_rgba(base: np.ndarray, overlay: np.ndarray, center: Tuple[float,float]):
    bx, by = int(center[0]), int(center[1])
    oh, ow = overlay.shape[:2]
    x1 = bx - ow//2; y1 = by - oh//2
    x2 = x1 + ow; y2 = y1 + oh
    X1 = max(0, x1); Y1 = max(0, y1)
    X2 = min(base.shape[1], x2); Y2 = min(base.shape[0], y2)
    if X1 >= X2 or Y1 >= Y2:
        return base
    ox1, oy1 = X1 - x1, Y1 - y1
    ox2, oy2 = ox1 + (X2 - X1), oy1 + (Y2 - Y1)
    crop = overlay[oy1:oy2, ox1:ox2]
    if crop.shape[2] < 4:
        base[Y1:Y2, X1:X2] = crop[..., :3]
        return base
    alpha = crop[..., 3:4] / 255.0
    for c in range(3):
        base[Y1:Y2, X1:X2, c] = (1.0 - alpha[...,0]) * base[Y1:Y2, X1:X2, c] + alpha[...,0] * crop[..., c]
    return base

# ---------------------------
# OBB ヘルパー
# ---------------------------
def edge_list_from_pts(pts: np.ndarray):
    edges = []
    for i in range(4):
        p1 = pts[i]; p2 = pts[(i+1)%4]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        angle = math.degrees(math.atan2(vec[1], vec[0]))
        mid = (p1 + p2) / 2.0
        edges.append({"i":i, "p1":p1, "p2":p2, "vec":vec, "len":length, "angle":angle, "mid":mid})
    return edges

def short_edge_indices(edges):
    lens = [e["len"] for e in edges]
    idx = np.argsort(lens)
    return [int(idx[0]), int(idx[1])]

def long_edge_indices(edges):
    lens = [e["len"] for e in edges]
    idx = np.argsort(lens)
    return [int(idx[-1]), int(idx[-2])]

def midpoint_of_edge(edges, idx):
    return edges[idx]["mid"]

def distance_point_to_segment(p, a, b):
    pa = p - a; ba = b - a
    denom = np.dot(ba, ba) + 1e-12
    t = np.dot(pa, ba) / denom
    t = max(0.0, min(1.0, t))
    proj = a + t * ba
    return np.linalg.norm(p - proj)

# ---------------------------
# 要素配置（テンプレ向き差分をここで吸収）
# ---------------------------
def compute_placement_for_element(pts_raw, box_xywhr, cls_name, beam_lines):
    pts = np.array(pts_raw, dtype=float)
    # normalize pts order for stable behavior
    pts = order_pts_clockwise_start(pts, start_hint="top-left")
    edges = edge_list_from_pts(pts)
    short_idxs = short_edge_indices(edges)
    long_idxs = long_edge_indices(edges)
    center = pts.mean(axis=0)

    # nearest beam
    nearest_beam = None
    min_d = 1e9
    for b in beam_lines:
        d = distance_point_to_segment(center, b["a"], b["b"])
        if d < min_d:
            min_d = d; nearest_beam = b

    result = {"class": cls_name, "center": center, "angle": 0.0, "scale": 1.0, "pts": pts}

    # beam
    if "beam" in cls_name:
        m1 = midpoint_of_edge(edges, short_idxs[0])
        m2 = midpoint_of_edge(edges, short_idxs[1])
        v = m2 - m1
        if np.linalg.norm(v) < 1e-6:
            v = np.array([1.0,0.0])
        angle_deg = math.degrees(math.atan2(v[1], v[0]))
        angle_deg = round(angle_deg / 15.0) * 15.0
        angle_deg = (angle_deg + 180) % 360
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        centerline_mid = (m1 + m2) / 2.0
        result.update({"center": centerline_mid, "angle": angle_deg, "scale": long_len, "a": m1, "b": m2})
        return result

    # pin / roller
    if cls_name in ("pin", "roller"):
        if nearest_beam is not None:
            beam_mid = (nearest_beam["a"] + nearest_beam["b"]) / 2.0
            desired_vec = beam_mid - center
            if np.linalg.norm(desired_vec) < 1e-6:
                _,_,_,_,ang = box_xywhr
                angle_deg = -math.degrees(ang)
            else:
                angle_deg = math.degrees(math.atan2(desired_vec[1], desired_vec[0]))
            avg_short = (edges[short_idxs[0]]["len"] + edges[short_idxs[1]]["len"]) / 2.0
            shift_dir = desired_vec / (np.linalg.norm(desired_vec) + 1e-12)
            centre_shifted = center + shift_dir * (avg_short * 0.02)
        else:
            _,_,_,_,ang = box_xywhr
            angle_deg = -math.degrees(ang)
            avg_short = (edges[short_idxs[0]]["len"] + edges[short_idxs[1]]["len"]) / 2.0
            centre_shifted = center

        # Keep angle info for debug, but user asked: "テンプレのまま角度関係なく貼る"
        # So we return angle candidate as detected, but overlay will ignore rotation for pin/roller.
        angle_for_debug = ((angle_deg + 180.0) % 360.0) - 180.0
        result.update({"center": centre_shifted, "angle": angle_for_debug, "scale": avg_short})
        return result

    # fixed support
    if cls_name in ("fixed", "fix", "kotei"):
        m1 = midpoint_of_edge(edges, short_idxs[0])
        m2 = midpoint_of_edge(edges, short_idxs[1])
        v = m2 - m1
        if np.linalg.norm(v) < 1e-6:
            v = np.array([1.0, 0.0])
        angle_deg = math.degrees(math.atan2(v[1], v[0]))
        angle_for_template = ((angle_deg + 180.0) % 360.0) - 180.0
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        centerline_mid = (m1 + m2) / 2.0
        result.update({"center": centerline_mid, "angle": angle_for_template, "scale": long_len})
        return result

    # load (keep existing behavior but keep normalization)
    if cls_name in ("load", "kajyu"):
        m1 = midpoint_of_edge(edges, short_idxs[0])
        m2 = midpoint_of_edge(edges, short_idxs[1])
        seg_mid = (m1 + m2) / 2.0
        shaft = m2 - m1
        if nearest_beam is not None:
            d1 = distance_point_to_segment(m1, nearest_beam["a"], nearest_beam["b"])
            d2 = distance_point_to_segment(m2, nearest_beam["a"], nearest_beam["b"])
            tip = m1 if d1 < d2 else m2
            tail = m2 if d1 < d2 else m1
        else:
            tip, tail = m2, m1
        dir_final = tip - tail
        if np.linalg.norm(dir_final) < 1e-6:
            dir_final = shaft
        angle_deg = math.degrees(math.atan2(dir_final[1], dir_final[0]))
        # keep the +180 correction you used earlier (you said load is currently reversed)
        angle_deg = (angle_deg + 180.0) % 360.0
        angle_deg = angle_deg if angle_deg <= 180.0 else angle_deg - 360.0
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        result.update({"center": seg_mid, "angle": angle_deg, "scale": long_len, "tip": tip})
        return result

    # udl
    if cls_name == "udl":
        if nearest_beam is not None:
            beam_dir = nearest_beam["dir_unit"]
            best_edge = None; best_dot = -1.0
            for e in edges:
                ev = e["vec"] / (np.linalg.norm(e["vec"]) + 1e-12)
                dot = abs(np.dot(ev, beam_dir))
                if dot > best_dot:
                    best_dot = dot; best_edge = e
            angle_deg = best_edge["angle"]
        else:
            angle_deg = edges[long_idxs[0]]["angle"]
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        result.update({"center": center, "angle": angle_deg, "scale": long_len})
        return result

    # fallback
    long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
    angle_deg = edges[long_idxs[0]]["angle"]
    result.update({"center": center, "angle": angle_deg, "scale": long_len})
    return result

# ---------------------------
# --- ここから節点関連ユーティリティ
# ---------------------------
def cluster_points(points: List[np.ndarray], merge_thresh: float) -> List[np.ndarray]:
    """
    単純な距離閾値クラスタリング（連結成分的マージ）。
    points: list of (2,) numpy arrays
    """
    if not points:
        return []
    pts = np.array(points, dtype=float)
    N = len(pts)
    groups = [{i} for i in range(N)]
    # union-find by distance
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                if groups[i] is None or groups[j] is None:
                    continue
                # pick any representative
                rep_i = next(iter(groups[i]))
                rep_j = next(iter(groups[j]))
                if np.linalg.norm(pts[rep_i] - pts[rep_j]) <= merge_thresh:
                    # merge j into i
                    groups[i] = groups[i].union(groups[j])
                    groups[j] = None
                    changed = True
    # build centroids
    clusters = []
    for g in groups:
        if g is None:
            continue
        idxs = sorted(list(g))
        centroid = pts[idxs].mean(axis=0)
        clusters.append(centroid)
    return clusters

def find_nearest_node_id(pt: np.ndarray, nodes: List[np.ndarray]) -> int:
    if not nodes:
        return -1
    dists = [np.linalg.norm(pt - n) for n in nodes]
    return int(np.argmin(dists))

def build_nodes_from_geometry(beam_lines, supports, loads, img_diag):
    """
    beam_lines: [{'a':[x,y], 'b':[x,y], 'dir_unit':...}, ...]
    supports:   [{'center':[x,y], 'class':'pin'|'roller'|'fixed'}, ...]
    loads:      [{'tip':[x,y], 'center':[x,y]}, ...]
    """

    # =========================
    # 1. 支点ノードをまず作る
    # =========================
    support_nodes = []

    # 上方向ベクトル（画像では y が下に向かうので -1）
    up = np.array([0, -1], dtype=float)

    # 支点処理（ピン・ローラーは上方向、固定は梁へ投影）
    for s in supports:
        center = np.array(s["center"], dtype=float)
        cls = s["class"].lower()

        # ------------------------
        # pin / roller → 上端ノード
        # ------------------------
        if cls in ("pin", "roller"):
            offset = max(10.0, img_diag * 0.015)  # 画像に対する適度な距離
            node = center + up * offset
            support_nodes.append(node)

        # ------------------------
        # fixed → 最も近い梁に合わせる
        # ------------------------
        elif cls == "fixed":
            closest_pt = None
            closest_dist = 1e9

            # 梁のいずれかに投影
            for b in beam_lines:
                a = np.array(b["a"], dtype=float)
                d = np.array(b["dir_unit"], dtype=float)  # 単位方向ベクトル

                # center を梁直線へ投影
                t = np.dot(center - a, d)
                proj = a + t * d

                dist = np.linalg.norm(center - proj)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_pt = proj

            # 投影点を支点ノードに採用
            if closest_pt is not None:
                support_nodes.append(closest_pt)
            else:
                # fallback：とりあえず中心
                support_nodes.append(center)

    # =========================
    # 2. 梁端点も候補に入れる
    # =========================
    beam_nodes = []
    for b in beam_lines:
        beam_nodes.append(np.array(b["a"], dtype=float))
        beam_nodes.append(np.array(b["b"], dtype=float))

    # =========================
    # 3. 荷重も先端を優先して追加
    # =========================
    load_nodes = []
    for L in loads:
        if "tip" in L and L["tip"] is not None:
            load_nodes.append(np.array(L["tip"], dtype=float))
        else:
            load_nodes.append(np.array(L["center"], dtype=float))

    # =========================
    # 4. すべてまとめてクラスタリング
    # =========================
    candidates = support_nodes + beam_nodes + load_nodes

    merge_thresh = max(8.0, img_diag * 0.01)  # しっかり統合する距離
    clustered = cluster_points(candidates, merge_thresh)

    # numpy 配列に変換して返す
    nodes = [np.array(c, dtype=float) for c in clustered]
    return nodes

# ---------------------------
# Main
# ---------------------------
def main():
    st.title("🏗️ 構造図 清書アプリ（fixed-v5: 節点検出付き）")
    st.write("・ピン/ローラーはテンプレの向きそのまま貼る（角度無視）")
    model_path = st.text_input("YOLO OBB model path", value=DEFAULT_MODEL_PATH)
    conf_th = st.sidebar.slider("検出信頼度", 0.0, 1.0, 0.5, 0.01)
    show_det = st.sidebar.checkbox("検出ポリゴン表示", value=True)
    show_cleaned = st.sidebar.checkbox("清書画像表示", value=True)
    show_nodes = st.sidebar.checkbox("節点表示 (番号付き)", value=True)
    uploaded = st.file_uploader("構造図アップロード", type=["png","jpg","jpeg"])
    template_files = {
        "pin": os.path.join(TEMPLATE_DIR, "pin.png"),
        "roller": os.path.join(TEMPLATE_DIR, "roller.png"),
        "fixed": os.path.join(TEMPLATE_DIR, "fixed.png"),
        "beam": os.path.join(TEMPLATE_DIR, "beam.png"),
        "load": os.path.join(TEMPLATE_DIR, "load.png"),
        "momentl": os.path.join(TEMPLATE_DIR, "momentL.png"),
        "momentr": os.path.join(TEMPLATE_DIR, "momentR.png"),
        "udl": os.path.join(TEMPLATE_DIR, "UDL.png"),
        "hinge": os.path.join(TEMPLATE_DIR, "hinge.png"),
    }
    TEMPL = {k: load_template_rgba(v) for k,v in template_files.items()}

    model = None
    if model_path and os.path.exists(model_path):
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"モデルロード失敗: {e}")
            model = None

    if uploaded is None:
        st.info("画像アップロードしてね")
        return

    try:
        img_pil = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("画像読み込み失敗")
        return

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    st.image(img_pil, caption="入力画像", use_container_width=True)

    if model is None:
        st.warning("モデル未ロード")
        return

    if not st.button("実行"):
        return

    with st.spinner("推論中..."):
        try:
            res = model(img, conf=conf_th, imgsz=640)[0]
        except Exception as e:
            st.error(f"推論失敗: {e}")
            return

    vis = img.copy()
    cleaned = np.ones_like(img) * 255
    beam_lines = []
    elements = []

    # detection
    if hasattr(res, "obb") and res.obb:
        obb = res.obb
        N = len(to_numpy(obb.xyxyxyxy))
        for i in range(N):
            conf = float(to_numpy(obb.conf[i]))
            if conf < conf_th:
                continue
            cls_id = int(to_numpy(obb.cls[i]))
            name = res.names[cls_id].lower().replace(" ", "")

            try:
                pts = to_numpy(obb.xyxyxyxy[i]).reshape(4,2)
            except Exception:
                continue

            # make vertex order stable (CW + top-left start)
            pts = order_pts_clockwise_start(pts, start_hint="top-left")

            try:
                xywhr = to_numpy(obb.xywhr[i])
                box_xywhr = tuple(map(float, xywhr))
            except Exception:
                box_xywhr = (0,0,0,0,0)

            edges = edge_list_from_pts(pts)
            if "beam" in name:
                sidx = short_edge_indices(edges)
                m1 = midpoint_of_edge(edges, sidx[0])
                m2 = midpoint_of_edge(edges, sidx[1])
                dir_unit = (m2 - m1) / (np.linalg.norm(m2 - m1) + 1e-12)
                beam_lines.append({"a": m1, "b": m2, "dir_unit": dir_unit, "pts": pts})

            elements.append({"name": name, "pts": pts, "box_xywhr": box_xywhr, "conf": conf})

            if show_det:
                pts_i = pts.astype(int)
                cv2.polylines(vis, [pts_i], True, (0,255,0), 2)
                for vi, p in enumerate(pts_i):
                    cv2.circle(vis, tuple(p), 5, (0,0,255), -1)
                    cv2.putText(vis, str(vi+1), (p[0]+5, p[1]+5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if show_det:
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="検出表示", use_container_width=True)

    # placement
    placements = []
    supports = []  # 支点リスト（pin/roller/fixed)
    loads = []     # 荷重リスト (load)
    for e in elements:
        try:
            place = compute_placement_for_element(e["pts"], e["box_xywhr"], e["name"], beam_lines)
            placements.append((e, place))
            if e["name"] in ("pin", "roller", "fixed"):
                supports.append({"class": e["name"], "center": place["center"], "angle": place.get("angle", 0.0)})
            if e["name"] in ("load", "kajyu"):
                loads.append({"class": e["name"], "center": place["center"], "tip": place.get("tip", None), "angle": place.get("angle", 0.0)})
        except Exception as ex:
            st.write(f"配置計算失敗: {e['name']} : {ex}")

    # overlay (テンプレ貼り)
    for elem, place in placements:
        name = elem["name"]
        tpl = TEMPL.get(name)
        center = place["center"]
        angle = float(place["angle"])  # angle kept for debug
        raw_scale = float(place.get("scale", 1.0))

        # prepare factor
        th, tw = (tpl.shape[0], tpl.shape[1]) if tpl is not None else (0,0)
        tpl_long = max(th, tw) if tpl is not None else 40
        if raw_scale > 10:
            factor = max(raw_scale / tpl_long, 0.35)
        else:
            factor = max(raw_scale / 40.0, 0.35)

        if tpl is None:
            cv2.circle(cleaned, (int(center[0]), int(center[1])), 6, (0,0,255), -1)
            continue

        if "beam" in name:
            factor = max(raw_scale / tpl_long, 0.5)
            tpl_scaled = scale_image(tpl, factor)
            angle_use = round(angle / 15.0) * 15.0
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle_use)
            cleaned = overlay_rgba(cleaned, tpl_rot, center)

        elif name in ("pin", "roller"):
            tpl_scaled = scale_image(tpl, factor)
            cleaned = overlay_rgba(cleaned, tpl_scaled, center)

        elif name == "fixed":
            tpl_scaled = scale_image(tpl, factor)
            angle_use = round(angle / 15.0) * 15.0
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle_use)
            cleaned = overlay_rgba(cleaned, tpl_rot, center)

        else:
            tpl_scaled = scale_image(tpl, factor)
            angle_use = round(angle / 15.0) * 15.0
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle_use)
            cleaned = overlay_rgba(cleaned, tpl_rot, center)

    # ========== 節点生成 ==========
    img_diag = math.hypot(img.shape[0], img.shape[1])
    nodes = build_nodes_from_geometry(beam_lines, supports, loads, img_diag)

    # build elements from beams: map beam endpoints to nearest nodes (avoid duplicates)
    elements_conn = []
    for b in beam_lines:
        a = np.array(b["a"], dtype=float)
        bb = np.array(b["b"], dtype=float)
        ida = find_nearest_node_id(a, nodes)
        idb = find_nearest_node_id(bb, nodes)
        if ida == idb:
            # degenerate: try to ignore or skip
            continue
        elements_conn.append({"type": "beam", "n1": ida, "n2": idb, "a": a.tolist(), "b": bb.tolist()})

    # attach supports to nearest node
    supports_attached = []
    for s in supports:
        nid = find_nearest_node_id(np.array(s["center"], dtype=float), nodes)
        supports_attached.append({"type": s["class"], "node": nid, "center": s["center"].tolist(), "angle": float(s.get("angle",0.0))})

    # attach loads to nearest node (prefer tip if exists)
    loads_attached = []
    for L in loads:
        pick = L.get("tip", None)
        if pick is None:
            pick = L["center"]
        nid = find_nearest_node_id(np.array(pick, dtype=float), nodes)
        loads_attached.append({"type": L["class"], "node": nid, "center": L["center"].tolist(), "tip": (L.get("tip").tolist() if L.get("tip") is not None else None), "angle": float(L.get("angle",0.0))})

    # draw nodes on cleaned and vis for debug
    node_img = cleaned.copy()
    vis_nodes_img = vis.copy()
    for idx, n in enumerate(nodes):
        x,y = int(n[0]), int(n[1])
        cv2.circle(node_img, (x,y), 6, (255,0,0), -1)
        cv2.putText(node_img, f"N{idx}", (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        cv2.circle(vis_nodes_img, (x,y), 6, (255,0,0), -1)
        cv2.putText(vis_nodes_img, f"N{idx}", (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    # also draw elements connections on vis for clarity
    for elem_conn in elements_conn:
        n1 = elem_conn["n1"]; n2 = elem_conn["n2"]
        p1 = tuple(map(int, nodes[n1]))
        p2 = tuple(map(int, nodes[n2]))
        cv2.line(vis_nodes_img, p1, p2, (0,128,255), 2)
        cv2.line(node_img, p1, p2, (0,128,255), 2)

    if show_cleaned:
        if show_nodes:
            st.image(cv2.cvtColor(node_img, cv2.COLOR_BGR2RGB), caption="清書結果（節点表示）", use_container_width=True)
        else:
            st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), caption="清書結果", use_container_width=True)

    if show_det:
        st.image(cv2.cvtColor(vis_nodes_img, cv2.COLOR_BGR2RGB), caption="検出（節点/要素接続表示）", use_container_width=True)

    # ========== 出力: 表 & JSON ==========
    st.subheader("節点一覧")
    node_table = []
    for i, n in enumerate(nodes):
        node_table.append({"id": i, "x": float(n[0]), "y": float(n[1])})
    st.table(node_table)

    st.subheader("要素（梁）一覧（節点インデックス）")
    st.table([{"idx": i, "n1": e["n1"], "n2": e["n2"]} for i,e in enumerate(elements_conn)])

    st.subheader("支持条件と荷重（節点インデックスへアタッチ済み）")
    st.write("supports:")
    st.table(supports_attached)
    st.write("loads:")
    st.table(loads_attached)

    # JSON ダウンロード用
    model_struct = {
        "nodes": [{"id": i, "x": float(n[0]), "y": float(n[1])} for i,n in enumerate(nodes)],
        "elements": elements_conn,
        "supports": supports_attached,
        "loads": loads_attached,
        "meta": {"image_shape": [int(img.shape[0]), int(img.shape[1])]}
    }
    json_str = json.dumps(model_struct, indent=2, ensure_ascii=False)
    st.download_button("構造データ(JSON)をダウンロード", data=json_str, file_name="structure_nodes.json", mime="application/json")

    st.subheader("配置デバッグ")
    for elem, place in placements:
        c = place["center"]
        st.write(f"{elem['name']}  center=({c[0]:.1f}, {c[1]:.1f})  angle={place['angle']:.1f}")

if __name__ == "__main__":
    main()
