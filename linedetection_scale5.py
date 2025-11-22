import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 紙サイズの実寸（mm）
PAPER_SIZES = {
    "A3": (420, 297),
    "A4": (297, 210),
    "B4": (364, 257),
    "B5": (257, 182)
}

st.title("構造図から応力図を自動生成するツール（スケール推定付き）")

uploaded_file = st.file_uploader("構造図の画像をアップロードしてください", type=["jpg", "jpeg", "png"])
selected_paper = st.selectbox("構造図を描いた紙サイズを選択してください", list(PAPER_SIZES.keys()))

if uploaded_file and selected_paper:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    paper_width_mm, paper_height_mm = PAPER_SIZES[selected_paper]

    # 画像のアスペクト比を計算
    img_aspect = width / height
    paper_aspect = paper_width_mm / paper_height_mm

    # アスペクト比を元に紙が画像内で縦向きか横向きか推定
    if abs(img_aspect - paper_aspect) < 0.2:
        # アスペクト比が近い＝紙がフレームいっぱいに写っていると仮定
        mm_per_pixel = paper_width_mm / width  # 横幅基準
    else:
        # 違いが大きければ、縦方向基準で仮定
        mm_per_pixel = paper_height_mm / height

    # m単位に変換
    m_per_pixel = mm_per_pixel / 1000
    L = width * m_per_pixel

    st.write(f"推定スケール: {mm_per_pixel:.3f} mm/pixel（紙サイズと画像の比から推定）")
    st.write(f"推定梁の長さ: {L:.3f} m")

    # エッジ検出と線分検出
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    # 荷重（仮に中央に10N）
    P = 10
    RA = P / 2
    RB = P / 2

    # 応力図プロット
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

    # 線分の描画
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="検出された構造線", use_column_width=True)
    else:
        st.warning("線分が検出されませんでした。画像が不鮮明な可能性があります。")

