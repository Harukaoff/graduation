import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title("梁（最長直線）と支点（三角形）の検出アプリ")

uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # === 二値化 + エッジ検出 ===
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, 50, 150)

    # === 最も長い直線を梁とする ===
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    longest_line = None
    max_len = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > max_len:
                max_len = length
                longest_line = (x1, y1, x2, y2)

    output = image_np.copy()
    if longest_line:
        x1, y1, x2, y2 = longest_line
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 3)  # 赤：梁

    # === 支点（三角形）検出 ===
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3 and cv2.isContourConvex(approx):
            cv2.drawContours(output, [approx], -1, (0, 0, 255), 2)  # 赤枠で三角形（調整必要なら青に）

    st.image(output, caption="検出結果（赤=梁・支点）", channels="RGB")
    st.caption("※ 梁：最長の直線、 支点：三角形と判定された輪郭")
