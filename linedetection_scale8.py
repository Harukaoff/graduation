import cv2
import numpy as np
import streamlit as st

# --- 関数：三角形（三支点）の検出 ---
def detect_triangles(image_gray, min_area=100, epsilon_ratio=0.02):
    triangles = []
    contours, _ = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)
        if len(approx) == 3 and area > min_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                triangles.append((cx, cy))
    return triangles

# --- 関数：線分を分類 ---
def classify_lines(lines, triangles, image_shape):
    height, width = image_shape
    beam_lines = []
    edge_lines = []
    other_lines = []

    for x1, y1, x2, y2 in lines:
        # 梁：横線、端に三角形が少なくとも1つ
        if abs(y1 - y2) < 10:
            for tx, ty in triangles:
                if np.hypot(x1 - tx, y1 - ty) < 20 or np.hypot(x2 - tx, y2 - ty) < 20:
                    beam_lines.append((x1, y1, x2, y2))
                    break
            else:
                other_lines.append((x1, y1, x2, y2))

        # 紙の端：画像の端に近く、縦 or 横線
        elif (min(x1, x2) < 10 or max(x1, x2) > width - 10 or
              min(y1, y2) < 10 or max(y1, y2) > height - 10):
            edge_lines.append((x1, y1, x2, y2))

        else:
            other_lines.append((x1, y1, x2, y2))

    return beam_lines, edge_lines, other_lines

# --- 関数：画像から線分を抽出 ---
def detect_lines(image_gray):
    edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        return [line[0] for line in lines]
    return []

# --- Main処理 ---
st.title("構造線分と紙端の分類ツール")
uploaded_file = st.file_uploader("構造図の画像をアップロード", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_display = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # 三角形検出
    triangles = detect_triangles(binary)
    for cx, cy in triangles:
        cv2.circle(image_display, (cx, cy), 5, (0, 0, 255), -1)

    # 線分検出
    lines = detect_lines(gray)
    beam_lines, edge_lines, other_lines = classify_lines(lines, triangles, gray.shape)

    # 線の描画
    for x1, y1, x2, y2 in beam_lines:
        cv2.line(image_display, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青 = 梁
    for x1, y1, x2, y2 in edge_lines:
        cv2.line(image_display, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 緑 = 紙の縁
    for x1, y1, x2, y2 in other_lines:
        cv2.line(image_display, (x1, y1), (x2, y2), (0, 0, 0), 1)    # その他は黒

    st.image(cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB), caption="分類された線分", use_column_width=True)

    st.markdown(f"**検出された梁の本数:** {len(beam_lines)}")
    st.markdown(f"**検出された紙の縁の本数:** {len(edge_lines)}")
    st.markdown(f"**検出されたその他の線:** {len(other_lines)}")
