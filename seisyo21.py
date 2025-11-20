# seisyo_final_fixed.py
import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from typing import List, Tuple, Dict

st.set_page_config(layout="wide", page_title="構造図 清書 + 可視化")

# ---------------------------
# 設定（デフォルトモデル・テンプレフォルダ）
# ---------------------------
DEFAULT_MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
TEMPLATE_DIR = "templates"

# ---------------------------
# ヘルパー: テンプレ読み込み（RGBA確保）
# ---------------------------
def load_template_rgba(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        b,g,r = cv2.split(img)
        a = np.ones_like(b) * 255
        img = cv2.merge([b,g,r,a])
    return img

# ---------------------------
# 回転・スケール・アルファ合成ユーティリティ
# - rotate_image_keep_alpha: CCW 角度（deg）
# - scale_image: factor
# - overlay_rgba: overlay の center (x,y) に配置して合成
# ---------------------------
def rotate_image_keep_alpha(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w/2.0, h/2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0,2] += (nw/2.0 - cx)
    M[1,2] += (nh/2.0 - cy)
    rotated = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def scale_image(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        return img
    h, w = img.shape[:2]
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def overlay_rgba(base: np.ndarray, overlay: np.ndarray, center: Tuple[float,float]):
    bx, by = int(center[0]), int(center[1])
    h_, w_ = overlay.shape[:2]
    x1 = bx - w_//2
    y1 = by - h_//2
    x2 = x1 + w_
    y2 = y1 + h_
    X1 = max(0, x1); Y1 = max(0, y1)
    X2 = min(base.shape[1], x2); Y2 = min(base.shape[0], y2)
    if X1 >= X2 or Y1 >= Y2:
        return base
    ox1, oy1 = X1 - x1, Y1 - y1
    ox2, oy2 = ox1 + (X2 - X1), oy1 + (Y2 - Y1)
    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    if overlay_crop.shape[2] < 4:
        # no alpha, direct paste
        base[Y1:Y2, X1:X2] = overlay_crop[..., :3]
        return base
    alpha = overlay_crop[..., 3:4] / 255.0
    for c in range(3):
        base[Y1:Y2, X1:X2, c] = (1 - alpha[...,0]) * base[Y1:Y2, X1:X2, c] + alpha[...,0] * overlay_crop[..., c]
    return base

# ---------------------------
# OBB 頂点順序・幾何関数（clockwise）
# ---------------------------
def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    order = np.argsort(ang)   # CCW ascending
    pts_ccw = pts[order]
    pts_cw = pts_ccw[::-1]    # clockwise
    return pts_cw

def edge_list_from_pts(pts: np.ndarray):
    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        angle = math.degrees(math.atan2(vec[1], vec[0]))
        mid = (p1 + p2) / 2.0
        edges.append({"i": i, "p1": p1, "p2": p2, "vec": vec, "len": length, "angle": angle, "mid": mid})
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

def line_from_two_points(a, b):
    v = b - a
    norm = np.linalg.norm(v) + 1e-12
    return a, v / norm

def distance_point_to_segment(p, a, b):
    pa = p - a
    ba = b - a
    denom = np.dot(ba, ba) + 1e-12
    t = np.dot(pa, ba) / denom
    t = max(0.0, min(1.0, t))
    proj = a + t * ba
    return np.linalg.norm(p - proj)

# ---------------------------
# 再配置ルール（beam / pin / roller / fixed / load / udl）
# 重要: テンプレは「右向き（tip＝右）」を前提
# ---------------------------
def compute_placement_for_element(pts_raw: np.ndarray, cls_name: str, beam_lines: List[Dict]):
    pts = order_points_clockwise(np.array(pts_raw, dtype=float))
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

    # ---- BEAM: 短辺中点を結ぶ centerline, 15°スナップ, scale = long_len ----
    if "beam" in cls_name:
        m1 = midpoint_of_edge(edges, short_idxs[0])
        m2 = midpoint_of_edge(edges, short_idxs[1])
        a, dir_unit = line_from_two_points(m1, m2)
        angle_deg = math.degrees(math.atan2(dir_unit[1], dir_unit[0]))
        angle_deg = round(angle_deg / 15) * 15
        long_len = max(edges[long_idxs[0]]['len'], edges[long_idxs[1]]['len'])
        centerline_mid = (m1 + m2) / 2.0
        result.update({"center": centerline_mid, "angle": angle_deg, "scale": long_len, "a": m1, "b": m2})
        return result

    # ---- PIN / ROLLER: tip (右向きテンプレ) が beam に接するように角度決定 ----
    if cls_name in ("pin", "roller"):
        if nearest_beam is not None:
            beam_dir = nearest_beam["dir_unit"]
            # choose box edge most parallel to beam (we treat that side as "down/toward beam")
            best_edge = None; best_dot = -1.0
            for e in edges:
                ev = e["vec"] / (np.linalg.norm(e["vec"]) + 1e-12)
                dot = abs(np.dot(ev, beam_dir))
                if dot > best_dot:
                    best_dot = dot; best_edge = e
            ev = best_edge["vec"] / (np.linalg.norm(best_edge["vec"]) + 1e-12)
            # normal pointing from center toward that edge
            normal = np.array([-ev[1], ev[0]])
            if np.dot(normal, best_edge["mid"] - center) < 0:
                normal = -normal
            # we want template's tip (right direction) to point toward beam.
            # direction from template center to beam = -normal (pointing toward beam)
            desired_dir = -normal
            angle_deg = math.degrees(math.atan2(desired_dir[1], desired_dir[0]))
            avg_short = (edges[short_idxs[0]]["len"] + edges[short_idxs[1]]["len"]) / 2.0
            # place slightly shifted outward so tip touches beam better
            result.update({"center": center + (-normal)* (avg_short * 0.015), "angle": angle_deg, "scale": avg_short})
            return result
        else:
            # fallback: take topmost vertex as tip location assumption (rare)
            top_idx = int(np.argmin(pts[:,1]))
            tip = pts[top_idx]
            opp = pts[(top_idx+2)%4]
            angle_deg = math.degrees(math.atan2((opp-tip)[1], (opp-tip)[0]))
            avg_short = (edges[short_idxs[0]]["len"] + edges[short_idxs[1]]["len"]) / 2.0
            result.update({"center": center, "angle": angle_deg, "scale": avg_short})
            return result

    # ---- LOAD (矢印) ----
    if cls_name in ("load", "kajyu"):
        m1 = midpoint_of_edge(edges, short_idxs[0])
        m2 = midpoint_of_edge(edges, short_idxs[1])
        seg_mid = (m1 + m2) / 2.0
        shaft = m2 - m1
        if np.linalg.norm(shaft) < 1e-6:
            shaft = np.array([1.0, 0.0])
        # choose which midpoint is closer to beam -> that midpoint should be arrow head (tip)
        tip_point, tail_point = m1, m2
        if nearest_beam is not None:
            d1 = distance_point_to_segment(m1, nearest_beam["a"], nearest_beam["b"])
            d2 = distance_point_to_segment(m2, nearest_beam["a"], nearest_beam["b"])
            if d1 < d2:
                tip_point, tail_point = m1, m2
            else:
                tip_point, tail_point = m2, m1
        dir_final = tip_point - tail_point
        if np.linalg.norm(dir_final) < 1e-6:
            dir_final = shaft
        angle_deg = math.degrees(math.atan2(dir_final[1], dir_final[0]))
        long_len = max(edges[long_idxs[0]]['len'], edges[long_idxs[1]]['len'])
        result.update({"center": seg_mid, "angle": angle_deg, "scale": long_len, "tip": tip_point, "tail": tail_point})
        return result

    # ---- UDL ----
    if cls_name in ("udl",):
        if nearest_beam is not None:
            beam_dir = nearest_beam["dir_unit"]
            best_edge = None; best_dot = -1.0
            for e in edges:
                ev = e["vec"] / (np.linalg.norm(e["vec"])+1e-12)
                dot = abs(np.dot(ev, beam_dir))
                if dot > best_dot:
                    best_dot = dot; best_edge = e
            angle_deg = best_edge["angle"]
            long_len = max(edges[long_idxs[0]]['len'], edges[long_idxs[1]]['len'])
            result.update({"center": center, "angle": angle_deg, "scale": long_len})
            return result
        else:
            long_len = max(edges[long_idxs[0]]['len'], edges[long_idxs[1]]['len'])
            angle_deg = edges[long_idxs[0]]['angle']
            result.update({"center": center, "angle": angle_deg, "scale": long_len})
            return result

    # fixed fallback
    long_len = max(edges[long_idxs[0]]['len'], edges[long_idxs[1]]['len'])
    angle_deg = edges[long_idxs[0]]['angle']
    result.update({"center": center, "angle": angle_deg, "scale": long_len})
    return result

# ---------------------------
# UI & main flow
# ---------------------------
def main():
    st.title("🏗️ 構造図 清書アプリ（tip=右向きテンプレ前提）")
    st.write("・頂点は時計回りで番号表示。・beamは短辺中点結ぶ。・beamは15°刻みスナップ。")

    model_path = st.text_input("YOLO OBB model path", value=DEFAULT_MODEL_PATH)
    conf_th = st.sidebar.slider("検出信頼度 (conf)", 0.0, 1.0, 0.5, 0.01)
    show_detection = st.sidebar.checkbox("検出ポリゴン表示", value=True)
    show_cleaned = st.sidebar.checkbox("清書画像表示", value=True)
    uploaded = st.file_uploader("構造図をアップロード (png/jpg/jpeg)", type=["png","jpg","jpeg"])

    # load templates (keys must match names used in label/classes)
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
    TEMPLATES = {}
    for k, p in template_files.items():
        if os.path.exists(p):
            TEMPLATES[k] = load_template_rgba(p)
        else:
            TEMPLATES[k] = None

    model = None
    if model_path and os.path.exists(model_path):
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"モデルロード失敗: {e}")
            model = None
    else:
        st.warning("モデルファイルが見つかりません。パス確認してください。")

    if uploaded is None:
        st.info("まずは構造図画像をアップロードしてね。")
        return

    try:
        img_pil = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"画像読み込み失敗: {e}")
        return

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    st.image(img_pil, caption="入力画像", use_column_width=True)

    if model is None:
        st.warning("モデルが未ロードのため、検出はできません。モデルパスを指定して再実行してください。")
        return

    if st.button("実行: 検出 → 頂点順序 → 再配置 → 清書"):
        with st.spinner("モデル推論中..."):
            try:
                res = model(img, conf=conf_th, imgsz=640)[0]
            except Exception as e:
                st.error(f"推論失敗: {e}")
                return

        vis = img.copy()
        cleaned = np.ones_like(img) * 255
        beam_lines = []
        elements = []

        if hasattr(res, "obb") and res.obb is not None:
            obb = res.obb
            N = obb.xyxyxyxy.shape[0] if hasattr(obb.xyxyxyxy, "shape") else len(obb.xyxyxyxy)
            for i in range(N):
                try:
                    conf = float(obb.conf[i].cpu().numpy()) if hasattr(obb.conf[i], "cpu") else float(obb.conf[i])
                    cls_id = int(obb.cls[i].cpu().numpy()) if hasattr(obb.cls[i], "cpu") else int(obb.cls[i])
                except Exception:
                    continue
                if conf < conf_th:
                    continue
                name = res.names[cls_id].lower().replace(" ", "")
                try:
                    xy8 = obb.xyxyxyxy[i].cpu().numpy().reshape(4,2)
                except Exception:
                    continue
                pts = order_points_clockwise(xy8)
                edges = edge_list_from_pts(pts)
                if "beam" in name:
                    sidx = short_edge_indices(edges)
                    m1 = midpoint_of_edge(edges, sidx[0])
                    m2 = midpoint_of_edge(edges, sidx[1])
                    dir_unit = (m2 - m1) / (np.linalg.norm(m2 - m1) + 1e-12)
                    beam_lines.append({"a": m1, "b": m2, "dir_unit": dir_unit, "pts": pts})
                elements.append({"name": name, "pts": pts, "conf": conf, "raw_index": i})

                if show_detection:
                    pts_i = pts.astype(np.int32)
                    cv2.polylines(vis, [pts_i], True, (0, 255, 0), 2)
                    for vi, p in enumerate(pts_i):
                        cv2.circle(vis, tuple(p.tolist()), 4, (0,0,255), -1)
                        # 頂点番号は時計回りで 1..4
                        cv2.putText(vis, str(vi+1), (int(p[0]+6), int(p[1]+6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cx = int(pts[:,0].mean()); cy = int(pts[:,1].mean())
                    cv2.putText(vis, f"{name} {conf:.2f}", (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        else:
            st.warning("OBB が見つかりませんでした。")
            return

        if show_detection:
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="検出可視化（頂点番号は時計回り）", use_column_width=True)

        # compute placements
        placements = []
        for elem in elements:
            try:
                place = compute_placement_for_element(elem["pts"], elem["name"], beam_lines)
            except Exception as e:
                st.write(f"配置計算失敗 {elem['name']}: {e}")
                continue
            placements.append((elem, place))

        # overlay templates
        for elem, place in placements:
            name = elem["name"]
            tpl = TEMPLATES.get(name)
            center = place["center"]
            angle = float(place["angle"])
            raw_scale = float(place.get("scale", 1.0))

            if tpl is None:
                cx, cy = int(center[0]), int(center[1])
                cv2.circle(cleaned, (cx,cy), 5, (0,0,255), -1)
                continue

            th, tw = tpl.shape[:2]
            tpl_long = max(tw, th)
            # raw_scale: pixel-length (long_len) -> factor
            if raw_scale > 10:
                factor = max(raw_scale / tpl_long, 0.35)
            else:
                factor = max(raw_scale / 40.0, 0.35)

            # beams: snap angle to 15°, scale with long edge
            if "beam" in name:
                angle = round(angle / 15) * 15
                factor = max(raw_scale / tpl_long, 0.5)
                tpl_scaled = scale_image(tpl, factor)
                tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle)
                cleaned = overlay_rgba(cleaned, tpl_rot, center)

            elif name in ("pin", "roller", "fixed"):
                # template tip defaults to right-center (because B pattern)
                tpl_scaled = scale_image(tpl, factor)
                h0, w0 = tpl_scaled.shape[:2]
                tip_px_scaled = (w0-1, h0//2)  # right-most center
                # rotate template
                tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle)
                # compute rotated tip position
                cx0, cy0 = w0/2.0, h0/2.0
                M = cv2.getRotationMatrix2D((cx0, cy0), angle, 1.0)
                tip_h = np.array([tip_px_scaled[0], tip_px_scaled[1], 1.0])
                tip_rot = M.dot(tip_h)
                tpl_rot_h, tpl_rot_w = tpl_rot.shape[:2]
                tip_rot[0] += (tpl_rot_w/2.0 - cx0)
                tip_rot[1] += (tpl_rot_h/2.0 - cy0)
                # we want tip_rot -> place at desired center (which compute_placement gave as contact point)
                desired = center
                shift_x = int(desired[0] - tip_rot[0])
                shift_y = int(desired[1] - tip_rot[1])
                overlay_center = (shift_x + tpl_rot_w/2.0, shift_y + tpl_rot_h/2.0)
                cleaned = overlay_rgba(cleaned, tpl_rot, overlay_center)

            else:
                # generic (load, udl, moment, hinge)
                tpl_scaled = scale_image(tpl, factor)
                tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle)
                cleaned = overlay_rgba(cleaned, tpl_rot, center)
                # if load, put red dot at tip for debug
                if "load" in name and "tip" in place:
                    tip = place["tip"]
                    cv2.circle(cleaned, (int(tip[0]), int(tip[1])), 6, (0,0,255), -1)

        if show_cleaned:
            st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), caption="清書結果 (テンプレ貼付)", use_column_width=True)

        st.subheader("配置一覧（デバッグ）")
        for elem, place in placements:
            c = place["center"]
            st.write(f"{elem['name']} conf={elem['conf']:.2f} center=({c[0]:.1f},{c[1]:.1f}) angle={place['angle']:.1f} scale_raw={place.get('scale',0):.1f}")

if __name__ == "__main__":
    main()
