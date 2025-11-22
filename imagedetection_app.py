import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# タイトル
st.title("構造図から応力図を自動生成するツール")

# 画像アップロード
uploaded_file = st.file_uploader("構造図の画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # エッジ検出
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 線分検出
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

    # 梁の長さ仮定（仮の数値データを使って計算）
    L = 10  # m（任意の仮定）
    P = 10  # N（中央に加わる荷重）
    
    # 反力計算
    RA = P * 0.5
    RB = P * 0.5

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

    # 元画像の表示（線検出）
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="検出された線分付き画像", use_column_width=True)
    else:
        st.warning("線分が検出されませんでした。より鮮明な画像を試してください。")
