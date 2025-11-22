import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.title("構造図の清書：水平梁の一本化＋三角形検出")

uploaded_file = st.file_uploader(
    "構造図画像をアップロードしてください",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # パラメータ
    min_area = st.sidebar.slider("三角形最小面積", 100, 5000, 400)
    approx_epsilon = st.sidebar.slider("近似精度", 0.01, 0.2, 0.04)
    block_size = st.sidebar.slider("adaptiveThreshold blockSize", 3, 61, 21, step=2)
    c = st.sidebar.slider("adaptiveThreshold C", 0, 20, 6)
    canny1 = st.sidebar.slider("Cannyしきい値1", 1, 200, 50)
    canny2 = st.sidebar.slider("Cannyしきい値2", 1, 200, 150)

    hough_thresh = st.sidebar.slider("HoughLinesP threshold", 10, 300, 100)
    min_line_len = st.sidebar.slider("minLineLength", 10, 300, 80)
    max_line_gap = st.sidebar.slider("maxLineGap", 1, 50, 10)
    y_threshold = 30  # ←固定値として設定

    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, c)
    edges = cv2.Canny(binary, canny1, canny2)

    st.subheader("エッジ検出結果")
    st.image(edges, clamp=True)

    # 三角形検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    triangle_mask = np.zeros_like(binary)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        epsilon = approx_epsilon * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            triangles.append(approx)
            cv2.drawContours(triangle_mask, [approx], -1, 255, -1)

    st.write(f"検出された三角形数: {len(triangles)}")

    # 三角形除外
    edges_no_tri = cv2.bitwise_and(edges, cv2.bitwise_not(triangle_mask))

    # 直線検出
    lines = cv2.HoughLinesP(edges_no_tri, 1, np.pi / 180, hough_thresh,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 10 or angle > 170:
                horizontal_lines.append(line)

    # グループ化関数（y座標30以内）
    def group_by_y(lines, threshold=30):
        groups = []
        used = [False] * len(lines)
        for i, line1 in enumerate(lines):
            if used[i]:
                continue
            x1, y1, x2, y2 = line1[0]
            y_mean = (y1 + y2) / 2
            group = [line1]
            used[i] = True
            for j, line2 in enumerate(lines):
                if used[j]:
                    continue
                x3, y3, x4, y4 = line2[0]
                y_mean2 = (y3 + y4) / 2
                if abs(y_mean - y_mean2) <= threshold:
                    group.append(line2)
                    used[j] = True
            groups.append(group)
        return groups

    # 近似直線を描画する
    def draw_fit_line(group, img, color=(255, 0, 0), thickness=3, idx=0):
        all_pts = []
        x_all = []
        for line in group:
            x1, y1, x2, y2 = line[0]
            all_pts.append([x1, y1])
            all_pts.append([x2, y2])
            x_all.extend([x1, x2])
        pts = np.array(all_pts)
        fit = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = fit.flatten()
        x_min, x_max = min(x_all), max(x_all)
        if abs(vx) < 1e-5:
            return
        slope = vy / vx
        intercept = y0 - slope * x0
        y_start = int(slope * x_min + intercept)
        y_end = int(slope * x_max + intercept)
        cv2.line(img, (x_min, y_start), (x_max, y_end), color, thickness)

    # 出力画像
    output = np.full_like(image, 255)

    # 三角形描画（赤）
    for triangle in triangles:
        cv2.drawContours(output, [triangle], -1, (0, 0, 255), -1)

    # 水平直線グループ化して一本化描画（青）
    if horizontal_lines:
        groups = group_by_y(horizontal_lines, threshold=y_threshold)
        for idx, group in enumerate(groups):
            draw_fit_line(group, output, color=(255, 0, 0), thickness=3, idx=idx)

    # 表示
    st.subheader("清書構造図（赤：三角形、青：梁）")
    st.image(output, channels="BGR")

    # ダウンロード
    _, buf = cv2.imencode(".png", output)
    st.download_button("清書画像をダウンロード", buf.tobytes(), "cleaned_structure.png", "image/png")
