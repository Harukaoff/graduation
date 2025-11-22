import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("構造図から 梁（直線）と 支点（三角形）を検出")

uploaded_file = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # === 2値化処理（黒画素ベース）===
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 輪郭抽出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image_np.copy()
    st.image(binary, caption="2値化画像（黒部分を検出）", channels="GRAY")

    # === 梁と支点の検出 ===
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        # 外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1e-5)

        # 梁：アスペクト比が大きい長い矩形
        if aspect > 5:
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 緑

        # 支点：三角形近似
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            cv2.drawContours(output, [approx], -1, (255, 0, 0), 2)  # 青

    st.image(output, caption="検出結果（緑=梁、青=支点）", channels="RGB")

