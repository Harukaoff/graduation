import streamlit as st
import cv2
import numpy as np

st.title("構造図と方眼紙の線を検出するツール")

uploaded_file = st.file_uploader("構造図の画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # ==== コントラスト強調 ====
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # ==== HSV変換で方眼紙検出（薄い青）====
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_grid = np.array([85, 10, 130])
    upper_grid = np.array([140, 255, 255])
    grid_mask = cv2.inRange(hsv, lower_grid, upper_grid)

    # 方眼の線分検出
    edges_grid = cv2.Canny(grid_mask, 50, 150)
    lines_grid = cv2.HoughLinesP(edges_grid, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=5)

    # ==== 構造線（黒線）検出 ====
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_black = cv2.Canny(gray, 50, 150)
    lines_black = cv2.HoughLinesP(edges_black, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10)

    # ==== 描画用コピー ====
    img_result = img.copy()

    # 方眼線を緑で描画
    if lines_grid is not None:
        for line in lines_grid:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_result, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 構造線を赤で描画
    if lines_black is not None:
        for line in lines_black:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # ==== 表示 ====
    st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), caption="検出された線分", use_column_width=True)
    st.image(grid_mask, caption="方眼マスク画像（白が検出された領域）", use_column_width=True)
