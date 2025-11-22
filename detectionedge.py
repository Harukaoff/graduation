import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# タイトル
st.title("構造図から応力図要素を検出・表示するツール")

# 画像アップロード
uploaded_file = st.file_uploader("構造図の画像をアップロード", type=["jpg", "jpeg", "png"])

# トリミングマージン（0〜1の割合）
top_margin = st.slider("上端トリミング", 0.0, 0.5, 0.05)
bottom_margin = st.slider("下端トリミング", 0.0, 0.5, 0.05)
left_margin = st.slider("左端トリミング", 0.0, 0.5, 0.05)
right_margin = st.slider("右端トリミング", 0.0, 0.5, 0.05)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    h, w = img.shape[:2]

    # マージンに基づくトリミング
    y1, y2 = int(h * top_margin), int(h * (1 - bottom_margin))
    x1, x2 = int(w * left_margin), int(w * (1 - right_margin))
    img_trimmed = img[y1:y2, x1:x2]
    st.image(img_trimmed, caption="トリミング後の画像", use_column_width=True)

    gray = cv2.cvtColor(img_trimmed, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # ハフ変換による直線検出（梁検出）
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

    # カラーで描画用画像をコピー
    result_img = img_trimmed.copy()
    
    # 梁の描画（緑）
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # 横線とみなす
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 荷重（矢印）と支点（三角形）検出用に輪郭抽出
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:  # 三角形：支点と仮定
            cv2.drawContours(result_img, [approx], -1, (255, 0, 0), 2)
        elif h > w and h > 20:  # 縦長：下向き矢印と仮定（荷重）
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    st.image(result_img, caption="要素検出結果（梁=緑, 荷重=赤, 支点=青）", use_column_width=True)

    # 水平位置スケール変換例（表示用）
    def get_position_ratio(x_pixel):
        return int(100 * x_pixel / (x2 - x1))

    st.write("例：左端0、右端100として変換したX座標を表示")
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:
                st.write(f"梁：{get_position_ratio(x1)} ~ {get_position_ratio(x2)}")
