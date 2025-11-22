import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

st.title("構造図の線分分類ツール")

uploaded_file = st.file_uploader("構造図をアップロード", type=["jpg", "jpeg", "png"])

def calculate_angle(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return abs(angle)  # 角度は絶対値で扱う（0-180°）

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # エッジ検出 → 線分検出
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=30, maxLineGap=10)

    if lines is None:
        st.error("線が検出できませんでした。画像を調整してください。")
    else:
        output = img.copy()
        h, w = gray.shape
        for line in lines:
            x1, y1, x2, y2 = line[0]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            length = math.hypot(x2 - x1, y2 - y1)
            angle = calculate_angle(x1, y1, x2, y2)

            # 分類ルール
            color = (0, 0, 255)  # default: 赤（未分類）

            if length > 0.8 * w or length > 0.8 * h:
                color = (0, 255, 255)  # 黄色: 紙の縁
            elif angle < 10 and mid_y > h * 0.5:
                color = (0, 255, 0)  # 緑: 梁（水平線）
            elif 80 < angle < 100 and mid_y < h * 0.5:
                color = (255, 0, 0)  # 青: 荷重矢印（縦線）

            cv2.line(output, (x1, y1), (x2, y2), color, 2)

        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                 caption="分類された線分", use_column_width=True)
