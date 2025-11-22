import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.title("構造図と方眼の線分検出ツール")

uploaded_file = st.file_uploader("構造図画像をアップロード (方眼紙に描かれた構造体)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert('RGB'))

    # 前処理（ぼかしなどでノイズを軽減）
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)

    # 色空間変換（青系と黒系の検出）
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)

    # 黒い構造体の線のマスク作成（低明度、低彩度）
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # 青い方眼線のマスク作成（青系の色相）
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 各マスクに対してエッジ検出と線分検出
    def detect_lines(mask, color):
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        line_img = np.zeros_like(image_np)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), color, 2)
        return line_img, lines

    structure_lines_img, structure_lines = detect_lines(mask_black, (0, 255, 0))  # 緑
    grid_lines_img, grid_lines = detect_lines(mask_blue, (255, 0, 0))  # 青

    # 元画像に重ね合わせ
    overlay_img = cv2.addWeighted(image_np, 0.8, structure_lines_img, 1.0, 0)
    overlay_img = cv2.addWeighted(overlay_img, 1.0, grid_lines_img, 1.0, 0)

    st.image(overlay_img, caption="検出結果（緑：構造線、青：方眼線）", use_column_width=True)

    st.markdown("---")
    if structure_lines is None:
        st.warning("構造体の黒い線が検出されませんでした。線が薄すぎるか、画像の明度が高すぎる可能性があります。")
    if grid_lines is None:
        st.warning("方眼の青い線が検出されませんでした。方眼が薄すぎるか、色味が異なる可能性があります。")