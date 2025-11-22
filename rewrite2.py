import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# テンプレート画像の読み込み
template_pin = cv2.imread('templates/pin2.png', cv2.IMREAD_UNCHANGED)
template_roller = cv2.imread('templates/roller2.png', cv2.IMREAD_UNCHANGED)
template_fixed = cv2.imread('templates/fixed1.png', cv2.IMREAD_UNCHANGED)

def paste_template(bg, template, center_x, center_y, scale=1.0):
    h, w = template.shape[:2]
    resized = cv2.resize(template, (int(w * scale), int(h * scale)))
    th, tw = resized.shape[:2]
    x1, y1 = int(center_x - tw // 2), int(center_y - th // 2)
    x2, y2 = x1 + tw, y1 + th
    if resized.shape[2] == 4:
        alpha = resized[:, :, 3] / 255.0
        for c in range(3):
            bg[y1:y2, x1:x2, c] = (1 - alpha) * bg[y1:y2, x1:x2, c] + alpha * resized[:, :, c]
    else:
        bg[y1:y2, x1:x2] = resized[:, :, :3]
    return bg

def draw_arrow(canvas, base, direction='down', length=80, color=(0, 0, 0), thickness=3):
    x, y = base
    tip = (x, y + length) if direction == 'down' else (x, y - length)
    return cv2.arrowedLine(canvas, base, tip, color, thickness, tipLength=0.5)

# Streamlit UI
st.title("構造体認識と再描画デモ")
uploaded_image = st.file_uploader("構造体の写真をアップロードしてください", type=['jpg', 'png'])

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 仮: 支点位置の自動認識処理（ここはテンプレートマッチング等で置き換え）
    detected_supports = [
        {"type": "pin", "pos": (200, 300)},
        {"type": "roller", "pos": (800, 300)}
    ]

    # 再描画
    canvas = np.ones_like(img) * 255  # 真っ白キャンバスに再描画
    canvas = cv2.line(canvas, (200, 300), (800, 300), (0, 0, 0), 8)  # 梁

    for support in detected_supports:
        if support["type"] == "pin":
            canvas = paste_template(canvas, template_pin, *support["pos"])
        elif support["type"] == "roller":
            canvas = paste_template(canvas, template_roller, *support["pos"])
        elif support["type"] == "fixed":
            canvas = paste_template(canvas, template_fixed, *support["pos"])

    canvas = draw_arrow(canvas, (500, 300))  # 荷重

    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="再描画された構造体", use_column_width=True)
