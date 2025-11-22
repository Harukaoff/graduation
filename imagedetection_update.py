import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 紙サイズの実寸（mm）
PAPER_SIZES = {
    "A3": (420, 297),
    "A4": (297, 210),
    "B4": (364, 257),
    "B5": (257, 182)
}

st.title("構造図から応力図を自動生成するツール")

uploaded_file = st.file_uploader("構造図の画像をアップロード", type=["jpg", "jpeg", "png"])
selected_paper = st.selectbox("紙サイズを選択してください", list(PAPER_SIZES.keys()))

def length_between_points(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

if uploaded_file is not None and selected_paper is not None:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    paper_width_mm, paper_height_mm = PAPER_SIZES[selected_paper]

    # スケール計算（幅方向の実寸をピクセル幅で割る）
    scale = paper_width_mm / width  # mm / pixel

    # ピクセル単位→m単位に変換するための係数
    scale_m = scale / 1000  # m / pixel

    # エッジ検出
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 輪郭検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arrow_lengths = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:  # 三角形（矢印の可能性）
            pts = approx.reshape(3, 2)
            centroid = pts.mean(axis=0)

            # 各頂点から重心までの距離を計算
            dists = [length_between_points(centroid, tuple(pt)) for pt in pts]
            max_dist_idx = np.argmax(dists)
            root_point = tuple(pts[max_dist_idx])
            tip_point = tuple(centroid.astype(int))

            length_px = length_between_points(root_point, tip_point)
            length_mm = length_px * scale
            arrow_lengths.append(length_mm)

    # 荷重Pの決定（10mmで5N換算）
    if arrow_lengths:
        arrow_length_mm = max(arrow_lengths)
        P = arrow_length_mm / 10 * 5  # N
    else:
        P = 10  # デフォルト荷重

    # 梁の長さ（m）
    L = width * scale_m

    # 反力計算（単純支持梁、中央荷重）
    RA = P * 0.5
    RB = P * 0.5

    # 線分検出（応力図に使うため）
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

    # 応力図描画
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))

    # せん断力図
    ax[0].plot([0, L/2, L], [RA, RA - P, 0], drawstyle='steps-post')
    ax[0].set_title("せん断力図")
    ax[0].set_xlabel("位置 (m)")
    ax[0].set_ylabel("せん断力 (N)")
    ax[0].grid(True)

    # 曲げモーメント図
    ax[1].plot([0, L/2, L], [0, -RA * (L/2), 0])
    ax[1].set_title("曲げモーメント図")
    ax[1].set_xlabel("位置 (m)")
    ax[1].set_ylabel("モーメント (Nm)")
    ax[1].grid(True)

    st.pyplot(fig)

    # 元画像に検出線を描画して表示
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="検出された線分付き画像", use_column_width=True)
    else:
        st.warning("線分が検出されませんでした。より鮮明な画像を試してください。")

