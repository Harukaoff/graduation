import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.title("構造図から応力図を自動生成するツール（方眼紙スケール対応）")

# 方眼紙のマスサイズ（mm）
grid_size_mm = st.number_input("方眼紙の1マスの大きさ（mm）を入力してください", value=10)

# 画像アップロード
uploaded_file = st.file_uploader("構造図の画像をアップロード", type=["jpg", "jpeg", "png"])

# 方眼スケール検出関数
def detect_grid_scale(gray_img, grid_size_mm):
    edges = cv2.Canny(gray_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=5)
    if lines is None:
        return None, None

    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 5:  # 横線
            horizontal_lines.append(line[0])
        elif abs(angle - 90) < 5 or abs(angle + 90) < 5:  # 縦線
            vertical_lines.append(line[0])

    vertical_x = sorted([(line[0] + line[2]) // 2 for line in vertical_lines])
    horizontal_y = sorted([(line[1] + line[3]) // 2 for line in horizontal_lines])

    vertical_diffs = [vertical_x[i+1] - vertical_x[i] for i in range(len(vertical_x) - 1)]
    horizontal_diffs = [horizontal_y[i+1] - horizontal_y[i] for i in range(len(horizontal_y) - 1)]

    mean_vert = np.mean(vertical_diffs) if vertical_diffs else None
    mean_horiz = np.mean(horizontal_diffs) if horizontal_diffs else None

    if mean_vert and mean_horiz:
        mean_pixel = (mean_vert + mean_horiz) / 2
        scale = grid_size_mm / mean_pixel  # mm/pixel
        return scale, lines
    else:
        return None, lines

# メイン処理
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scale, lines = detect_grid_scale(gray, grid_size_mm)

    if scale is None:
        st.warning("スケールが検出できませんでした。方眼紙が鮮明に写る画像を使用してください。")
    else:
        st.success(f"スケール検出成功！ 1ピクセル = {scale:.4f} mm")

        # 梁の長さを画像幅で仮定（mm→m）
        L = img.shape[1] * scale / 1000  # 幅 × スケール（mm/pix） → m

        # 荷重仮定
        P = 10  # N（中央荷重）
        RA = P * 0.5
        RB = P * 0.5

        # 応力図描画
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))

        # せん断力図
        ax[0].plot([0, L / 2, L], [RA, RA - P, 0], drawstyle='steps-post')
        ax[0].set_title("せん断力図")
        ax[0].set_xlabel("位置 (m)")
        ax[0].set_ylabel("せん断力 (N)")
        ax[0].grid(True)

        # モーメント図
        ax[1].plot([0, L / 2, L], [0, -RA * (L / 2), 0])
        ax[1].set_title("曲げモーメント図")
        ax[1].set_xlabel("位置 (m)")
        ax[1].set_ylabel("モーメント (Nm)")
        ax[1].grid(True)

        st.pyplot(fig)

        # 元画像に検出線を描画
        img_lines = img.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)
            st.image(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB), caption="検出された線付き画像", use_column_width=True)

