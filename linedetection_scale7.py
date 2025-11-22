import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def detect_edges_and_lines(image, paper_width_mm):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=50, maxLineGap=10)

    return edges, lines

def classify_lines(image, lines):
    height, width = image.shape[:2]
    classified = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)

        # 梁（太くて水平、黒、長さがある）
        if abs(y1 - y2) < 10 and length > width * 0.4:
            classified.append(((x1, y1, x2, y2), 'beam'))

        # 荷重（短い垂直線、梁の上あたり）
        elif abs(x1 - x2) < 10 and length < height * 0.4 and min(y1, y2) < height / 2:
            classified.append(((x1, y1, x2, y2), 'load'))

        else:
            classified.append(((x1, y1, x2, y2), 'other'))

    return classified

def draw_classified_lines(image, classified_lines):
    colors = {
        'beam': (0, 255, 0),    # 緑
        'load': (255, 0, 0),    # 青
        'other': (0, 0, 255)    # 赤
    }

    output = image.copy()
    for ((x1, y1, x2, y2), label) in classified_lines:
        cv2.line(output, (x1, y1), (x2, y2), colors[label], 2)
        cv2.putText(output, label, ((x1 + x2)//2, (y1 + y2)//2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 1)

    return output

st.title("構造図線分分類ツール")

uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])
paper_width_mm = st.number_input("使用した紙の横幅 (mm)", min_value=100, max_value=500, value=210)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption="アップロードされた画像", use_column_width=True)

    edges, lines = detect_edges_and_lines(image, paper_width_mm)

    if lines is not None:
        classified_lines = classify_lines(image, lines)
        output = draw_classified_lines(image, classified_lines)
        st.image(output, caption="分類結果", use_column_width=True)
    else:
        st.warning("線分が検出できませんでした。画像のコントラストや解像度を確認してください。")