import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile

st.title("📐 構造図から応力図を自動生成するツール（基準物付きスケール認識）")
st.markdown("""
- 画像内に**10cmの定規の一部**を必ず入れてください（横か縦の直線）。
- 定規のピクセル長を検出し、スケールを自動計算します。
- それを使って応力図を作成します。
""")

uploaded_file = st.file_uploader("📤 構造図の画像をアップロードしてください", type=["jpg", "jpeg", "png"])

def preprocess_image(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    return binary

def detect_lines(binary_img, min_length=30):
    edges = cv2.Canny(binary_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=min_length, maxLineGap=10)
    return lines

def find_scale_line(lines, image):
    if lines is None:
        return None
    candidate_lines = []
    height, width = image.shape[:2]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # 横または縦に近く、端に近い直線（定規を想定）
        if ((abs(angle) < 5 or abs(angle - 90) < 5) and
            (30 <= length <= 200) and
            (x1 < width * 0.3 or x2 < width * 0.3 or
             y1 > height * 0.7 or y2 > height * 0.7)):
            candidate_lines.append((line[0], length))
    if not candidate_lines:
        return None
    candidate_lines.sort(key=lambda x: x[1], reverse=True)
    return candidate_lines[0]

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img_path = tfile.name

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = preprocess_image(gray)
    st.image(binary, caption="前処理（二値化）画像", use_column_width=True)

    lines = detect_lines(binary)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)

    scale_line = find_scale_line(lines, img)

    if scale_line is None:
        st.error("基準物（10cmの定規の線）が検出できませんでした。画像に必ず入れてください。")
    else:
        (x1, y1, x2, y2), px_length = scale_line
        st.success(f"基準物の線の長さ（ピクセル）: {px_length:.1f}px")
        scale_m_per_px = 0.1 / px_length
        st.success(f"スケール: 1ピクセル = {scale_m_per_px:.5f} m")

        img_scale = img.copy()
        cv2.line(img_scale, (x1,y1), (x2,y2), (0,0,255), 3)
        st.image(cv2.cvtColor(img_scale, cv2.COLOR_BGR2RGB), caption="基準物の線を赤で強調表示", use_column_width=True)

        edges = cv2.Canny(gray, 50, 150)
        lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

        beam_line = None
        loads = []

        for line in lines2:
            x1_, y1_, x2_, y2_ = line[0]
            if abs(y2_ - y1_) < 10 and abs(x2_ - x1_) > 100:
                beam_line = (x1_, y1_, x2_, y2_)
            elif abs(x2_ - x1_) < 10 and 10 < abs(y2_ - y1_) < 100:
                cx = (x1_ + x2_) // 2
                cy = (y1_ + y2_) // 2
                length_px = abs(y2_ - y1_)
                loads.append((cx, cy, length_px))

        if beam_line is None:
            st.error("梁が検出できませんでした。画像を確認してください。")
        else:
            x1_, y1_, x2_, y2_ = beam_line
            beam_length_px = abs(x2_ - x1_)
            beam_length_m = beam_length_px * scale_m_per_px

            st.success(f"検出された梁の長さ: {beam_length_m:.2f} m")

            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title("検出された構造図（基準線赤）")
            st.pyplot(fig)

            x_vals = np.linspace(0, beam_length_m, int(beam_length_m * 100))
            V = np.zeros_like(x_vals)
            M = np.zeros_like(x_vals)

            for (cx, cy, length_px) in loads:
                load_pos_px = cx - min(x1_, x2_)
                load_pos_m = load_pos_px * scale_m_per_px
                load_val = length_px * 5

                st.write(f"🔻 荷重検出: 位置 = {load_pos_m:.2f} m, 強さ = {load_val:.2f} N")

                V[x_vals >= load_pos_m] -= load_val
                M[x_vals >= load_pos_m] -= load_val * (x_vals[x_vals >= load_pos_m] - load_pos_m)

            fig2, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))
            ax1.plot(x_vals, V, color='red')
            ax1.axhline(0, color='gray', linestyle='--')
            ax1.set_ylabel("せん断力 [N]")
            ax1.set_title("せん断力図")

            ax2.plot(x_vals, M, color='blue')
            ax2.axhline(0, color='gray', linestyle='--')
            ax2.set_xlabel("梁の位置 [m]")
            ax2.set_ylabel("曲げモーメント [Nm]")
            ax2.set_title("曲げモーメント図")

            st.pyplot(fig2)
