# seisyo_final_fixed_v4_connected_fixed.py
# 改良版：ピン/ローラーは回転無視、梁を支点で接続、荷重先端スナップ
import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from typing import Any, Tuple, List

st.set_page_config(layout="wide", page_title="構造図 清書 + 接続 (fixed-v4-connected-fix)")

# ---------------------------
# 設定
# ---------------------------
DEFAULT_MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
TEMPLATE_DIR = "templates"

# テンプレのデフォルト向き（テンプレ画像がどの方向を"tip"として持っているか）
# 例: 'up' = テンプレの先端が画像上方向を向いている
TEMPLATE_DEFAULT_TIP = {
    "pin": "up",
    "roller": "up",
    "fixed": "right",   # 固定支点は「直線が右向き」を想定
    "beam": "right",    # 梁テンプレは右向きに長辺が伸びる想定
    "load": "right",    # 矢印テンプレは右向きが矢印先端
    "momentl": "none",
    "momentr": "none",
    "udl": "right",
    "hinge": "none",
}

# スナップ閾値（ピクセル） — 必要に応じて調整
SNAP_DISTANCE_TO_BEAM = 100.0   # 支点候補が梁の近くとみなす距離
SNAP_NEAR_ENDPOINT = 40.0       # 支点が端点に近いとき端点に吸着
LOAD_SNAP_THRESHOLD = 100.0     # 荷重先端を梁先にスナップする閾値

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
# 角度ユーティリティ
# ---------------------------
def normalize_angle_deg(a: float) -> float:
    a = float(a) % 360.0
    if a > 180.0:
        a -= 360.0
    return a

# ---------------------------
# 頂点整列: CW + top-left start
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
# RGBA 読み込み・描画ユーティリティ
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
    if crop.size == 0:
        return base
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

def project_point_on_segment(p, a, b):
    pa = p - a; ba = b - a
    denom = np.dot(ba, ba) + 1e-12
    t = np.dot(pa, ba) / denom
    t_clamped = max(0.0, min(1.0, t))
    proj = a + t_clamped * ba
    return proj, t_clamped

# ---------------------------
# 要素配置（テンプレ向き差分をここで吸収）
# - pin/roller は角度を返すが、overlay 側で回転無視する（ユーザ指定）
# - beam は endpoints a,b を result に格納
# - load: tip を result に格納
# ---------------------------
def compute_placement_for_element(pts_raw, box_xywhr, cls_name, beam_lines):
    pts = np.array(pts_raw, dtype=float)
    pts = order_pts_clockwise_start(pts, start_hint="top-left")
    edges = edge_list_from_pts(pts)
    short_idxs = short_edge_indices(edges)
    long_idxs = long_edge_indices(edges)
    center = pts.mean(axis=0)

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
        # do NOT force +180 flip here; we'll define beam endpoints from node snapping
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

        # keep detected angle for debug, overlay will ignore rotation per user's request
        angle_for_debug = normalize_angle_deg(angle_deg)
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
        angle_for_template = normalize_angle_deg(angle_deg)
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        centerline_mid = (m1 + m2) / 2.0
        result.update({"center": centerline_mid, "angle": angle_for_template, "scale": long_len})
        return result

    # load / arrow
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
        # assume template arrow "tip is right", so template rotation = angle_deg
        angle_for_template = normalize_angle_deg(angle_deg)
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        result.update({"center": seg_mid, "angle": angle_for_template, "scale": long_len, "tip": tip, "tail": tail})
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
# 梁調整：支点（ノード）で梁を繋ぎ、各支点を梁上にスナップ
# - placements は (elem, place) のリストを直接更新する
# ---------------------------
def adjust_beams_by_nodes(placements: List[tuple]):
    node_names = set(["pin","roller","fixed","fix","kotei","hinge","load","kajyu","udl"])
    # separate beams and others
    beams = [(e,p) for e,p in placements if "beam" in e["name"]]
    others = [(e,p) for e,p in placements if "beam" not in e["name"]]

    for be, bplace in beams:
        a = np.array(bplace.get("a", bplace["center"]))
        b = np.array(bplace.get("b", bplace["center"]))
        ba = b - a
        denom = np.dot(ba, ba) + 1e-12
        candidates = []
        for oe, oplace in others:
            if oe["name"] not in node_names:
                continue
            p = np.array(oplace["center"])
            t = np.dot(p - a, ba) / denom
            proj = a + t * ba
            dist_to_line = np.linalg.norm(p - proj)
            # accept a reasonable strip; distance threshold tuned
            if -0.5 <= t <= 1.5 and dist_to_line < SNAP_DISTANCE_TO_BEAM:
                candidates.append({"elem": oe, "place": oplace, "p": p, "t": t, "dist": dist_to_line, "proj": proj})
        if len(candidates) >= 2:
            candidates_sorted = sorted(candidates, key=lambda x: x["t"])
            first = candidates_sorted[0]["proj"]
            last = candidates_sorted[-1]["proj"]
            # update beam endpoints & center & angle & scale
            angle_deg = math.degrees(math.atan2(last[1] - first[1], last[0] - first[0]))
            angle_deg = normalize_angle_deg(angle_deg)
            center = (first + last) / 2.0
            scale = float(np.linalg.norm(last - first))
            bplace["center"] = center
            bplace["angle"] = angle_deg
            bplace["scale"] = scale
            bplace["endpoint_a"] = first
            bplace["endpoint_b"] = last
            # snap each candidate node to beam projection
            for c in candidates_sorted:
                cplace = c["place"]
                cplace["center"] = c["proj"]
                cplace["snapped_to_beam"] = True
            # also snap nodes very near the endpoints to endpoints
            for oe, oplace in others:
                if oe["name"] not in node_names:
                    continue
                p = np.array(oplace["center"])
                if np.linalg.norm(p - first) < SNAP_NEAR_ENDPOINT:
                    oplace["center"] = first
                    oplace["snapped_to_beam"] = True
                if np.linalg.norm(p - last) < SNAP_NEAR_ENDPOINT:
                    oplace["center"] = last
                    oplace["snapped_to_beam"] = True
        else:
            # not enough candidates: keep original endpoints but still try to snap any nodes close to segment
            a = np.array(bplace.get("a", a))
            b = np.array(bplace.get("b", b))
            for oe, oplace in others:
                if oe["name"] not in node_names:
                    continue
                p = np.array(oplace["center"])
                proj, t = project_point_on_segment(p, a, b)
                if np.linalg.norm(p - proj) < SNAP_DISTANCE_TO_BEAM and 0.0 <= t <= 1.0:
                    oplace["center"] = proj
                    oplace["snapped_to_beam"] = True

# ---------------------------
# 荷重の先端を最も近い梁にスナップ
# ---------------------------
def snap_load_tips_to_beams(placements: List[tuple]):
    # collect beam segments (use endpoint if present)
    beams_seg = []
    for e, place in placements:
        if "beam" in e["name"]:
            if "endpoint_a" in place and "endpoint_b" in place:
                a = np.array(place["endpoint_a"])
                b = np.array(place["endpoint_b"])
            else:
                a = np.array(place.get("a", place["center"]))
                b = np.array(place.get("b", place["center"]))
            beams_seg.append({"a": a, "b": b, "place": place})
    # for each load, snap tip to closest beam projection
    for oe, oplace in placements:
        if oe["name"] in ("load", "kajyu"):
            tip = oplace.get("tip", None)
            if tip is None:
                continue
            tip = np.array(tip)
            best = None
            for bs in beams_seg:
                proj, t = project_point_on_segment(tip, bs["a"], bs["b"])
                d = np.linalg.norm(proj - tip)
                if best is None or d < best["d"]:
                    best = {"proj": proj, "d": d, "bs": bs}
            if best is not None and best["d"] < LOAD_SNAP_THRESHOLD:
                proj = best["proj"]
                cur_center = np.array(oplace["center"])
                # shift whole arrow so tip lands on proj
                delta = proj - tip
                new_center = cur_center + delta
                oplace["center"] = new_center
                oplace["tip_proj"] = proj
                oplace["snapped_to_beam"] = True

# ---------------------------
# Main
# ---------------------------
def main():
    st.title("🏗️ 構造図 清書アプリ（fixed-v4-connected-fix）")
    st.write("梁を支点で接続・荷重先端スナップ・ピン/ローラーは回転無視")

    model_path = st.text_input("YOLO OBB model path", value=DEFAULT_MODEL_PATH)
    conf_th = st.sidebar.slider("検出信頼度", 0.0, 1.0, 0.5, 0.01)
    show_det = st.sidebar.checkbox("検出ポリゴン表示", value=True)
    show_cleaned = st.sidebar.checkbox("清書画像表示", value=True)
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

    # detection & element list
    if hasattr(res, "obb") and res.obb:
        obb = res.obb
        N = len(to_numpy(obb.xyxyxyxy))
        for i in range(N):
            try:
                conf = float(to_numpy(obb.conf[i]))
            except Exception:
                continue
            if conf < conf_th:
                continue
            try:
                cls_id = int(to_numpy(obb.cls[i]))
            except Exception:
                continue
            name = res.names[cls_id].lower().replace(" ", "")

            try:
                pts = to_numpy(obb.xyxyxyxy[i]).reshape(4,2)
            except Exception:
                continue

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

    # initial placements
    placements = []
    for e in elements:
        try:
            place = compute_placement_for_element(e["pts"], e["box_xywhr"], e["name"], beam_lines)
            placements.append((e, place))
        except Exception as ex:
            st.write(f"配置計算失敗: {e['name']} : {ex}")

    # adjust beams by nodes (snap centers, compute endpoints)
    adjust_beams_by_nodes(placements)

    # snap loads' tips to beams
    snap_load_tips_to_beams(placements)

    # draw beam connectors (black thick lines) first
    connector_layer = np.zeros_like(cleaned)
    for e, place in placements:
        if "beam" in e["name"]:
            a = place.get("endpoint_a", None)
            b = place.get("endpoint_b", None)
            if a is None or b is None:
                a = place.get("a", place["center"])
                b = place.get("b", place["center"])
            a_i = (int(a[0]), int(a[1])); b_i = (int(b[0]), int(b[1]))
            cv2.line(connector_layer, a_i, b_i, (0,0,0), thickness=4, lineType=cv2.LINE_AA)
    mask = np.any(connector_layer != 0, axis=2)
    cleaned[mask] = connector_layer[mask]

    # overlay templates
    for elem, place in placements:
        name = elem["name"]
        tpl = TEMPL.get(name)
        center = place["center"]
        angle = float(place.get("angle", 0.0))
        raw_scale = float(place.get("scale", 1.0))

        if tpl is None:
            cv2.circle(cleaned, (int(center[0]), int(center[1])), 6, (0,0,255), -1)
            continue

        th, tw = tpl.shape[:2]
        tpl_long = max(th, tw)
        if raw_scale > 10:
            factor = max(raw_scale / tpl_long, 0.35)
        else:
            factor = max(raw_scale / 40.0, 0.35)

        if "beam" in name:
            factor = max(raw_scale / tpl_long, 0.5)
            tpl_scaled = scale_image(tpl, factor)
            angle_use = round(angle / 15.0) * 15.0
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle_use)
            cleaned = overlay_rgba(cleaned, tpl_rot, place["center"])

        elif name in ("pin", "roller"):
            tpl_scaled = scale_image(tpl, factor)
            # USER REQUEST: pin/roller keep template orientation (no rotation)
            cleaned = overlay_rgba(cleaned, tpl_scaled, place["center"])

        elif name == "fixed":
            tpl_scaled = scale_image(tpl, factor)
            angle_use = round(angle / 15.0) * 15.0
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle_use)
            cleaned = overlay_rgba(cleaned, tpl_rot, place["center"])

        elif name in ("load", "kajyu"):
            tpl_scaled = scale_image(tpl, factor)
            # Determine rotation according to template default tip direction
            tip_dir = TEMPLATE_DEFAULT_TIP.get(name, "right")
            if tip_dir == "right":
                angle_use = round(angle / 15.0) * 15.0
            elif tip_dir == "up":
                # template tip up -> subtract 90
                angle_use = round((angle - 90.0) / 15.0) * 15.0
            else:
                angle_use = round(angle / 15.0) * 15.0
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle_use)
            cleaned = overlay_rgba(cleaned, tpl_rot, place["center"])

        else:
            tpl_scaled = scale_image(tpl, factor)
            angle_use = round(angle / 15.0) * 15.0
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle_use)
            cleaned = overlay_rgba(cleaned, tpl_rot, place["center"])

    if show_cleaned:
        st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), caption="清書結果", use_container_width=True)

    st.subheader("配置デバッグ")
    for elem, place in placements:
        c = place["center"]
        ang_disp = place.get("angle", 0.0)
        snapped = place.get("snapped_to_beam", False)
        extra = " (snapped)" if snapped else ""
        st.write(f"{elem['name']}  center=({c[0]:.1f}, {c[1]:.1f})  angle={ang_disp:.1f}{extra}")

if __name__ == "__main__":
    main()
