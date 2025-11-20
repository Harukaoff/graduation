# seisyo_final_fixed_v2.py
import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from typing import List, Tuple, Dict

st.set_page_config(layout="wide", page_title="構造図 清書 + 可視化 (fixed-v2)")

# ---------------------------
# 設定
# ---------------------------
DEFAULT_MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
TEMPLATE_DIR = "templates"  # templates/pin.png 等を置く

# ---------------------------
# ヘルパー: RGBA テンプレ読み込み
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
# 画像回転・スケール・アルファ合成ユーティリティ
# ---------------------------
def rotate_image_keep_alpha(img: np.ndarray, angle_deg: float) -> np.ndarray:
    # CCW 角度
    h,w = img.shape[:2]
    cx, cy = w/2.0, h/2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
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
# OBB 頂点順序まわり（注意: ここでは「検出が返す頂点順」を保持する）
# - Roboflowラベル時と同じ順（left-top -> clockwise）を前提とする場合は
#   obb.xyxyxyxy がその順で来ているなら、そのまま扱う（再ソートしない）。
# - ただし、頂点の座標から各辺情報は作る。
# ---------------------------
def edge_list_from_pts(pts: np.ndarray):
    # pts shape (4,2) in the order provided by model (we DO NOT reorder)
    edges = []
    for i in range(4):
        p1 = pts[i]; p2 = pts[(i+1)%4]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        angle = math.degrees(math.atan2(vec[1], vec[0]))  # degrees CCW
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

def distance_point_to_segment(p, a, b):
    pa = p - a; ba = b - a
    denom = np.dot(ba, ba) + 1e-12
    t = np.dot(pa, ba) / denom
    t = max(0.0, min(1.0, t))
    proj = a + t * ba
    return np.linalg.norm(p - proj)

# ---------------------------
# 再配置ルール（改訂）
# - 重要: 支点系は「原図の向きを崩さない」→検出角度や頂点順は参照するが、
#   最終的な向きは”最寄りの梁へ向かうベクトル”を優先して決定する（tip を梁に合わせる）
# - 梁は短辺の中点を結ぶ centerline（15° 刻みでスナップ）
# ---------------------------
def compute_placement_for_element(pts_raw: np.ndarray, box_xywhr: Tuple[float,float,float,float,float], cls_name: str, beam_lines: List[Dict]):
    # pts_raw: (4,2) as given by model (we keep order)
    pts = np.array(pts_raw, dtype=float)
    edges = edge_list_from_pts(pts)
    short_idxs = short_edge_indices(edges)
    long_idxs = long_edge_indices(edges)
    center = pts.mean(axis=0)

    # find nearest beam to this element:
    nearest_beam = None
    min_d = 1e9
    for b in beam_lines:
        d = distance_point_to_segment(center, b["a"], b["b"])
        if d < min_d:
            min_d = d; nearest_beam = b

    # start result
    result = {"class": cls_name, "center": center, "angle": 0.0, "scale": 1.0, "pts": pts}

    # beam: centerline is midpoints of short edges
    if "beam" in cls_name:
        m1 = midpoint_of_edge(edges, short_idxs[0])
        m2 = midpoint_of_edge(edges, short_idxs[1])
        v = m2 - m1
        if np.linalg.norm(v) < 1e-6:
            v = np.array([1.0, 0.0])
        angle_deg = math.degrees(math.atan2(v[1], v[0]))
        angle_deg = round(angle_deg / 15) * 15  # snap 15°
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        centerline_mid = (m1 + m2) / 2.0
        result.update({"center": centerline_mid, "angle": angle_deg, "scale": long_len, "a": m1, "b": m2})
        return result

    # pin / roller: template は "右向き tip=右" を前提。だから tip を最寄り梁方向に向ける。
    if cls_name in ("pin", "roller"):
        if nearest_beam is not None:
            # desired direction: from element center toward nearest point on beam centerline
            beam_mid = (nearest_beam["a"] + nearest_beam["b"]) / 2.0
            desired_vec = beam_mid - center
            if np.linalg.norm(desired_vec) < 1e-6:
                # fallback: use box_xywhr angle if available
                _, _, _, _, ang = box_xywhr
                angle_deg = -math.degrees(ang)  # model gives rad CCW? we invert to match rotate_image CCW
            else:
                angle_deg = math.degrees(math.atan2(desired_vec[1], desired_vec[0]))
            # scale: use average of short edges
            avg_short = (edges[short_idxs[0]]["len"] + edges[short_idxs[1]]["len"]) / 2.0
            # shift slightly toward beam so tip touches better
            shift_dir = desired_vec / (np.linalg.norm(desired_vec) + 1e-12)
            result.update({"center": center + shift_dir * (avg_short * 0.02), "angle": angle_deg, "scale": avg_short})
            return result
        else:
            # no beam nearby: use box angle if present
            _, _, _, _, ang = box_xywhr
            angle_deg = -math.degrees(ang)
            avg_short = (edges[short_idxs[0]]["len"] + edges[short_idxs[1]]["len"]) / 2.0
            result.update({"center": center, "angle": angle_deg, "scale": avg_short})
            return result

    # load (arrow): shaft = connect midpoints of short edges; tip is the midpoint closer to beam
    if cls_name in ("load", "kajyu"):
        m1 = midpoint_of_edge(edges, short_idxs[0])
        m2 = midpoint_of_edge(edges, short_idxs[1])
        seg_mid = (m1 + m2) / 2.0
        shaft = m2 - m1
        if np.linalg.norm(shaft) < 1e-6:
            shaft = np.array([1.0, 0.0])
        # pick tip closer to beam if beam exists
        if nearest_beam is not None:
            d1 = distance_point_to_segment(m1, nearest_beam["a"], nearest_beam["b"])
            d2 = distance_point_to_segment(m2, nearest_beam["a"], nearest_beam["b"])
            tip = m1 if d1 < d2 else m2
            tail = m2 if d1 < d2 else m1
        else:
            # if no beam, assume arrow points from tail->tip = m1->m2 using labeling convention
            tip, tail = m2, m1
        dir_final = tip - tail
        if np.linalg.norm(dir_final) < 1e-6:
            dir_final = shaft
        angle_deg = math.degrees(math.atan2(dir_final[1], dir_final[0]))
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        result.update({"center": seg_mid, "angle": angle_deg, "scale": long_len, "tip": tip, "tail": tail})
        return result

    # udl: align left side of template to edge far from beam (approx)
    if cls_name in ("udl",):
        if nearest_beam is not None:
            beam_dir = nearest_beam["dir_unit"]
            best_edge = None; best_dot = -1.0
            for e in edges:
                ev = e["vec"] / (np.linalg.norm(e["vec"]) + 1e-12)
                dot = abs(np.dot(ev, beam_dir))
                if dot > best_dot:
                    best_dot = dot; best_edge = e
            angle_deg = best_edge["angle"]
            long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
            result.update({"center": center, "angle": angle_deg, "scale": long_len})
            return result
        else:
            long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
            angle_deg = edges[long_idxs[0]]["angle"]
            result.update({"center": center, "angle": angle_deg, "scale": long_len})
            return result

    # fixed fallback
    long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
    angle_deg = edges[long_idxs[0]]["angle"]
    result.update({"center": center, "angle": angle_deg, "scale": long_len})
    return result

# ---------------------------
# メイン UI / フロー
# ---------------------------
def main():
    st.title("🏗️ 構造図 清書アプリ（fixed-v2）")
    st.write("・検出（OBB）を尊重して支点を原図向きのまま配置。梁は短辺中点結ぶ／15° スナップ。")

    model_path = st.text_input("YOLO OBB model path", value=DEFAULT_MODEL_PATH)
    conf_th = st.sidebar.slider("検出信頼度 (conf)", 0.0, 1.0, 0.5, 0.01)
    show_detection = st.sidebar.checkbox("検出ポリゴン表示", value=True)
    show_cleaned = st.sidebar.checkbox("清書画像表示", value=True)
    uploaded = st.file_uploader("構造図アップロード (png/jpg/jpeg)", type=["png","jpg","jpeg"])

    # load templates
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
    TEMPLATES = {k: load_template_rgba(p) for k,p in template_files.items()}

    # load model if path valid
    model = None
    if model_path and os.path.exists(model_path):
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"モデルロード失敗: {e}")
            model = None
    else:
        st.warning("モデルが見つかりません。モデルパスを確認してください。")

    if uploaded is None:
        st.info("画像アップロードしてね。")
        return

    try:
        img_pil = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"画像読み込み失敗: {e}")
        return
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    st.image(img_pil, caption="入力画像", use_column_width=True)

    if model is None:
        st.warning("モデル未ロード。モデルパスを指定して実行してください。")
        return

    if not st.button("実行: 検出 → 再配置 → 清書"):
        return

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

    # iterate detections (robust)
    if hasattr(res, "obb") and res.obb is not None:
        obb = res.obb
        # number of boxes:
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
            # get 8points (model-provided polygon)
            try:
                xy8 = obb.xyxyxyxy[i].cpu().numpy().reshape(4,2)
            except Exception:
                # fallback: use axis-aligned xyxy if available
                try:
                    xyxy = obb.xyxy[i].cpu().numpy()
                    x1,y1,x2,y2 = map(int, xyxy)
                    xy8 = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=float)
                except Exception:
                    continue
            # get box xywhr if available
            try:
                xywhr = obb.xywhr[i].cpu().numpy()  # x,y,w,h,angle(rad)
                box_xywhr = tuple(map(float, xywhr))
            except Exception:
                box_xywhr = (0.0,0.0,0.0,0.0,0.0)
            pts = xy8  # keep order from model (Roboflow labeling order)
            edges = edge_list_from_pts(pts)
            if "beam" in name:
                sidx = short_edge_indices(edges)
                m1 = midpoint_of_edge(edges, sidx[0])
                m2 = midpoint_of_edge(edges, sidx[1])
                dir_unit = (m2 - m1) / (np.linalg.norm(m2 - m1) + 1e-12)
                beam_lines.append({"a": m1, "b": m2, "dir_unit": dir_unit, "pts": pts})
            elements.append({"name": name, "pts": pts, "conf": conf, "box_xywhr": box_xywhr, "raw_index": i})

            if show_detection:
                pts_i = pts.astype(np.int32)
                cv2.polylines(vis, [pts_i], True, (0,255,0), 2)
                # draw vertices numbered using model order (assumed left-top -> clockwise)
                for vi, p in enumerate(pts_i):
                    cv2.circle(vis, tuple(p.tolist()), 4, (0,0,255), -1)
                    cv2.putText(vis, str(vi+1), (int(p[0]+6), int(p[1]+6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cx = int(pts[:,0].mean()); cy = int(pts[:,1].mean())
                cv2.putText(vis, f"{name} {conf:.2f}", (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    else:
        st.warning("OBB 検出結果がありません。")
        return

    if show_detection:
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="検出可視化（モデル頂点順で番号表示）", use_column_width=True)

    # compute placements
    placements = []
    for e in elements:
        try:
            place = compute_placement_for_element(e["pts"], e["box_xywhr"], e["name"], beam_lines)
        except Exception as ex:
            st.write(f"配置計算失敗: {e['name']} : {ex}")
            continue
        placements.append((e, place))

    # overlay templates
    for elem, place in placements:
        name = elem["name"]
        tpl = TEMPLATES.get(name)
        center = place["center"]
        angle = float(place["angle"])
        raw_scale = float(place.get("scale", 1.0))

        if tpl is None:
            # draw debug dot
            cv2.circle(cleaned, (int(center[0]), int(center[1])), 6, (0,0,255), -1)
            continue

        th, tw = tpl.shape[:2]
        tpl_long = max(tw, th)
        # raw_scale likely pixel length (long_len) -> convert to factor
        if raw_scale > 10:
            factor = max(raw_scale / tpl_long, 0.35)
        else:
            factor = max(raw_scale / 40.0, 0.35)

        # beam: snap angle to 15
        if "beam" in name:
            angle = round(angle / 15) * 15
            factor = max(raw_scale / tpl_long, 0.5)
            tpl_scaled = scale_image(tpl, factor)
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle)
            cleaned = overlay_rgba(cleaned, tpl_rot, center)

        elif name in ("pin", "roller", "fixed"):
            # template assumed right-facing tip; we want tip to touch beam direction computed earlier
            tpl_scaled = scale_image(tpl, factor)
            h0, w0 = tpl_scaled.shape[:2]
            # tip relative location = right-middle (change if template differs)
            tip_px = (w0 - 1, h0 // 2)
            # rotate template
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle)
            # compute rotated tip position (affine center rotation)
            cx0, cy0 = w0/2.0, h0/2.0
            M = cv2.getRotationMatrix2D((cx0, cy0), angle, 1.0)
            tip_h = np.array([tip_px[0], tip_px[1], 1.0])
            tip_rot = M.dot(tip_h)
            tpl_rot_h, tpl_rot_w = tpl_rot.shape[:2]
            tip_rot[0] += (tpl_rot_w/2.0 - cx0)
            tip_rot[1] += (tpl_rot_h/2.0 - cy0)
            # desired contact point = place["center"]
            desired = center
            shift_x = int(desired[0] - tip_rot[0])
            shift_y = int(desired[1] - tip_rot[1])
            overlay_center = (shift_x + tpl_rot_w/2.0, shift_y + tpl_rot_h/2.0)
            cleaned = overlay_rgba(cleaned, tpl_rot, overlay_center)

        else:
            # generic overlay (load, udl, moment, hinge)
            tpl_scaled = scale_image(tpl, factor)
            tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle)
            cleaned = overlay_rgba(cleaned, tpl_rot, center)
            if "load" in name and "tip" in place:
                # show tip position for debugging (in detection coords)
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
