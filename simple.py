import cv2
import numpy as np
import math
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ======================
# 設定
# ======================
LOAD_SNAP_THRESHOLD = 40

# ======================
# テンプレ読み込み
# ======================
@st.cache_resource
def load_template(path):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if temp is None:
        st.error(f"テンプレート '{path}' が読み込めません")
    return temp

template_pin = load_template("templates/pin.png")
template_roller = load_template("templates/roller.png")
template_fixed = load_template("templates/fixed.png")
template_beam = load_template("templates/beam.png")
template_load = load_template("templates/load.png")

# ======================
# 画像へテンプレ貼り付け
# ======================
def paste(img, template, center):
    th, tw = template.shape[:2]
    x = int(center[0] - tw/2)
    y = int(center[1] - th/2)

    h, w = img.shape[:2]
    if x < 0 or y < 0 or x+tw > w or y+th > h:
        return img

    if template.shape[2] == 4:
        alpha = template[:, :, 3] / 255.0
        for c in range(3):
            img[y:y+th, x:x+tw, c] = (1 - alpha) * img[y:y+th, x:x+tw, c] + alpha * template[:, :, c]
    else:
        img[y:y+th, x:x+tw] = template

    return img

# ======================
# 梁を支点間に引く
# ======================
def draw_beam(img, p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = int(math.hypot(dx, dy))
    angle = math.degrees(math.atan2(dy, dx))

    # 梁テンプレを伸縮
    beam = cv2.resize(template_beam, (length, 20))
    M = cv2.getRotationMatrix2D((length/2, 10), angle, 1)
    beam_rot = cv2.warpAffine(beam, M, (length, 20),
                              flags=cv2.INTER_LINEAR,
                              borderValue=(0, 0, 0, 0))

    cx = (p1[0] + p2[0]) / 2
    cy = (p1[1] + p2[1]) / 2

    sx, sy = int(cx - length/2), int(cy - 10)
    h, w = img.shape[:2]

    if sx < 0 or sy < 0 or sx+length > w or sy+20 > h:
        return img

    alpha = beam_rot[:, :, 3] / 255.0
    for c in range(3):
        img[sy:sy+20, sx:sx+length, c] = \
            (1 - alpha) * img[sy:sy+20, sx:sx+length, c] + \
            alpha * beam_rot[:, :, c]

    return img

# ======================
# 荷重を梁にスナップ
# ======================
def snap_load_to_beam(load_pos, beams):
    min_dist = 9999
    snap_point = load_pos

    for (p1, p2) in beams:
        A = np.array(p1)
        B = np.array(p2)
        P = np.array(load_pos)
        AB = B - A
        t = np.dot(P - A, AB) / np.dot(AB, AB)
        t = max(0, min(1, t))
        proj = A + t * AB
        dist = np.linalg.norm(P - proj)

        if dist < min_dist:
            min_dist = dist
            snap_point = proj

    if min_dist < LOAD_SNAP_THRESHOLD:
        return int(snap_point[0]), int(snap_point[1])
    else:
        return load_pos

# ======================
# YOLO → 清書
# ======================
def overlay_all(img, results):
    supports = []
    loads = []
    beams = []

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        label = results.names[cls]

        if label in ["pin", "roller", "fixed"]:
            supports.append((label, (cx, cy)))

        elif label == "line":
            beams.append(((x1, y1), (x2, y2)))

        elif label == "load":
            loads.append((cx, cy))

    # 支点を貼る
    for typ, pos in supports:
        if typ == "pin": img = paste(img, template_pin, pos)
        elif typ == "roller": img = paste(img, template_roller, pos)
        elif typ == "fixed": img = paste(img, template_fixed, pos)

    # 支点同士を梁で結ぶ
    if len(supports) >= 2:
        for i in range(len(supports)-1):
            p1 = supports[i][1]
            p2 = supports[i+1][1]
            beams.append((p1, p2))
            img = draw_beam(img, p1, p2)

    # 荷重 → 最寄り梁へ
    for load_p in loads:
        sp = snap_load_to_beam(load_p, beams)
        img = paste(img, template_load, sp)

    return img

# ======================
# Streamlit UI
# ======================
def main():
    st.title("構造図 清書アプリ（超シンプル版）")

    uploaded = st.file_uploader("画像をアップロード", type=["png","jpg","jpeg"])
    model_file = st.file_uploader("YOLOモデル（.pt）を選択", type=["pt"])

    if uploaded and model_file:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)

        model = YOLO(model_file)

        results = model(img_np)[0]
        st.image(results.plot(), caption="検出結果", use_container_width=True)

        out = overlay_all(img_np.copy(), results)

        st.image(out, caption="清書結果", use_container_width=True)

        cv2.imwrite("output.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        st.success("保存しました (output.png)")

if __name__ == "__main__":
    main()
