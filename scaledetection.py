import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# 用紙サイズ（単位：m）
paper_sizes = {
    "A4": (0.210, 0.297),
    "B4": (0.257, 0.364),
    "B5": (0.182, 0.257)
}

st.title("📄 用紙サイズからスケールを自動認識する応力図生成ツール")
st.markdown("""
- A4, B4, B5の用紙を構造図と一緒に写してください。
- 用紙の長辺が写っている必要があります。
- ピクセル長と実寸を対応させてスケールを推定します。
""")

uploaded_file = st.file_uploader("📤 構造図の画像をアップロードしてください", type=["jpg", "jpeg", "png"])
paper_type = st.selectbox("📏 写っている用紙のサイズを選んでください", options=list(paper_sizes.keys()))

def preprocess_image(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged

def detect_paper_contour(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_contour = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_contour = approx
    return best_contour

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img_path = tfile.name

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = preprocess_image(gray)

    paper_contour = detect_paper_contour(edged)
    if paper_contour is None:
        st.error("用紙が検出できませんでした。白い背景に、はっきり写るようにしてください。")
    else:
        # 4点を取得して長辺を見つける
        pts = paper_contour.reshape(4, 2)
        dists = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
        long_edge_px = max(dists)
        long_edge_m = max(paper_sizes[paper_type])  # 長辺を使う

        scale_m_per_px = long_edge_m / long_edge_px
        st.success(f"検出された用紙の長辺（ピクセル）: {long_edge_px:.1f}px")
        st.success(f"スケール: 1ピクセル = {scale_m_per_px:.5f} m")

        # 用紙輪郭を描画
        img_contour = img.copy()
        cv2.drawContours(img_contour, [paper_contour], -1, (0, 255, 0), 3)
        st.image(cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB), caption="検出された用紙の輪郭", use_column_width=True)

        # 梁・荷重検出処理
        lines = cv2.HoughLinesP(edged, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        beam_line = None
        loads = []

        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                if abs(y2 - y1) < 10 and abs(x2 - x1) > 100:
                    beam_line = (x1, y1, x2, y2)
                elif abs(x2 - x1) < 10 and 10 < abs(y2 - y1) < 100:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    length_px = abs(y2 - y1)
                    loads.append((cx, cy, length_px))

        if beam_line is None:
            st.error("梁が検出できませんでした。構造図の撮影範囲やコントラストを確認してください。")
        else:
            x1_, y1_, x2_, y2_ = beam_line
            beam_length_px = abs(x2_ - x1_)
            beam_length_m = beam_length_px * scale_m_per_px

            st.success(f"検出された梁の長さ: {beam_length_m:.2f} m")

            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title("検出された構造図（用紙輪郭）")
            st.pyplot(fig)

            # 応力図生成
            x_vals = np.linspace(0, beam_length_m, int(beam_length_m * 100))
            V = np.zeros_like(x_vals)
            M = np.zeros_like(x_vals)

            for (cx, cy, length_px) in loads:
                load_pos_px = cx - min(x1_, x2_)
                load_pos_m = load_pos_px * scale_m_per_px
                load_val = length_px * 5  # 仮スケーリング

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
