import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.title("構造図画像から線分検出とスケール認識")

uploaded_file = st.file_uploader("構造図画像をアップロード", type=["jpg", "jpeg", "png"])
grid_size_mm = st.number_input("方眼1マスのサイズ（mm）", min_value=1, value=5)

if uploaded_file is not None:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # エッジ検出
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # ハフ変換で線分検出
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

    line_img = img.copy()

    if lines is not None:
        all_lengths_px = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            length_px = np.hypot(x2 - x1, y2 - y1)
            all_lengths_px.append(length_px)

        st.image(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB), caption="検出された線分", use_column_width=True)

        # スケール推定：1マスの方眼が最も多い長さと仮定
        hist, bins = np.histogram(all_lengths_px, bins=20)
        peak_index = np.argmax(hist)
        representative_length_px = (bins[peak_index] + bins[peak_index + 1]) / 2

        # スケール（1 px が何 m か）
        scale_m_per_px = (grid_size_mm / 1000) / representative_length_px
        st.markdown(f"**推定されたスケール：{scale_m_per_px:.5f} m/px**")

    else:
        st.warning("線分が検出されませんでした。画像が不鮮明な可能性があります。")
