import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================
# テンプレート読み込み関数
# ==========================
def load_templates(folder_path):
    templates = {}
    for name in ["pin", "roller", "fixed", "beam", "load", "moment l", "moment r", "udl"]:
        path = os.path.join(folder_path, f"{name}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            templates[name] = img
        else:
            templates[name] = None
    return templates

# ==========================
# テンプレート描画関数
# ==========================
def overlay_template(canvas, template, center, angle_deg, scale=1.0):
    if template is None:
        return canvas

    h, w = template.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(template, (new_w, new_h))

    M = cv2.getRotationMatrix2D((new_w/2, new_h/2), -angle_deg, 1.0)
    rotated = cv2.warpAffine(resized, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255,0))

    x, y = int(center[0] - new_w / 2), int(center[1] - new_h / 2)
    overlay = canvas.copy()
    for c in range(3):
        alpha = rotated[:,:,3] / 255.0
        overlay[y:y+new_h, x:x+new_w, c] = (1 - alpha) * overlay[y:y+new_h, x:x+new_w, c] + alpha * rotated[:,:,c]
    return overlay

# ==========================
# YOLO出力の解析
# ==========================
def parse_yolo_obb_results(results, conf_th=0.5):
    elems = []
    if not hasattr(results, "obb") or results.obb is None:
        return elems
    for box in results.obb:
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        if conf < conf_th:
            continue
        name = results.names[cls_id]
        x, y, w, h, angle = box.xywhr.cpu().numpy()[0]
        elems.append({
            "class": name,
            "x": float(x),
            "y": float(y),
            "w": float(w),
            "h": float(h),
            "angle": float(angle * 180 / np.pi),
            "conf": conf
        })
    return elems

def classify_elements(elems):
    supports = [e for e in elems if e["class"] in ["pin", "roller", "fixed"]]
    beams = [e for e in elems if e["class"] == "beam"]
    loads = [e for e in elems if e["class"] in ["load", "moment l", "moment r", "udl"]]
    return supports, beams, loads

# ==========================
# 梁と支点の接続
# ==========================
def connect_beams_to_supports(beams, supports, snap_thresh=40):
    connected = []
    for beam in beams:
        x, y, w, h, angle = beam["x"], beam["y"], beam["w"], beam["h"], beam["angle"]
        rad = np.deg2rad(angle)
        dx = (w / 2) * np.cos(rad)
        dy = (w / 2) * np.sin(rad)
        end1 = np.array([x - dx, y - dy])
        end2 = np.array([x + dx, y + dy])

        dists1 = [np.linalg.norm(np.array([s["x"], s["y"]]) - end1) for s in supports]
        dists2 = [np.linalg.norm(np.array([s["x"], s["y"]]) - end2) for s in supports]

        if len(supports) == 0:
            continue

        idx1, idx2 = np.argmin(dists1), np.argmin(dists2)
        s1, s2 = supports[idx1], supports[idx2]
        d1, d2 = dists1[idx1], dists2[idx2]

        if d1 < snap_thresh:
            end1 = np.array([s1["x"], s1["y"]])
        if d2 < snap_thresh:
            end2 = np.array([s2["x"], s2["y"]])

        connected.append({
            "beam": beam,
            "p1": tuple(end1),
            "p2": tuple(end2)
        })
    return connected

# ==========================
# 荷重を梁に沿って配置
# ==========================
def attach_loads_to_beams(loads, connected_beams):
    attached = []
    for load in loads:
        lx, ly = load["x"], load["y"]
        best_beam, best_proj, best_dist = None, None, 9999
        for b in connected_beams:
            p1 = np.array(b["p1"])
            p2 = np.array(b["p2"])
            v = p2 - p1
            t = np.dot(np.array([lx, ly]) - p1, v) / np.dot(v, v)
            t = np.clip(t, 0, 1)
            proj = p1 + t * v
            dist = np.linalg.norm(proj - np.array([lx, ly]))
            if dist < best_dist:
                best_dist, best_proj, best_beam = dist, proj, b
        if best_beam:
            load["x"], load["y"] = best_proj
            load["angle"] = best_beam["beam"]["angle"]
            attached.append(load)
    return attached

# ==========================
# 清書描画
# ==========================
def redraw_structure(base_img, supports, connected_beams, loads, templates):
    canvas = np.ones_like(base_img) * 255
    for b in connected_beams:
        beam_img = templates["beam"]
        bx = (b["p1"][0] + b["p2"][0]) / 2
        by = (b["p1"][1] + b["p2"][1]) / 2
        L = np.linalg.norm(np.array(b["p2"]) - np.array(b["p1"]))
        angle = b["beam"]["angle"]
        scale = L / beam_img.shape[1]
        canvas = overlay_template(canvas, beam_img, (bx, by), angle, scale)

    for s in supports:
        tname = s["class"]
        if tname in templates and templates[tname] is not None:
            canvas = overlay_template(canvas, templates[tname], (s["x"], s["y"]), 0, 1.0)

    for l in loads:
        tname = l["class"]
        if tname in templates and templates[tname] is not None:
            canvas = overlay_template(canvas, templates[tname], (l["x"], l["y"]), l["angle"], 1.0)
    return canvas

# ==========================
# Streamlit UI
# ==========================
st.title("📐 構造図 清書アプリ（YOLOv8-OBB + テンプレート配置）")

uploaded = st.file_uploader("構造図をアップロード", type=["png", "jpg", "jpeg"])

# モデルパス（修正版）
MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
TEMPLATES_PATH = "templates"

if uploaded:
    img_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="アップロード画像", use_column_width=True)

    with st.spinner("YOLOv8-OBBで解析中..."):
        model = YOLO(MODEL_PATH)
        results = model(img, conf=0.5, imgsz=640)[0]
        elems = parse_yolo_obb_results(results)
        supports, beams, loads = classify_elements(elems)

        connected_beams = connect_beams_to_supports(beams, supports)
        loads_attached = attach_loads_to_beams(loads, connected_beams)

        templates = load_templates(TEMPLATES_PATH)
        final_img = redraw_structure(img, supports, connected_beams, loads_attached, templates)

    st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption="清書結果", use_column_width=True)
