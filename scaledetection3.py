import cv2
import numpy as np
import streamlit as st

# 用紙サイズ（mm）辞書
PAPER_SIZES = {
    "A4": (210, 297),
    "B4": (257, 364),
    "B5": (182, 257)
}

def detect_paper_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def compute_scale(paper_corners, paper_size_mm):
    # 射影変換で正面から見た長方形に
    width_mm, height_mm = paper_size_mm
    paper_corners = order_points(paper_corners)
    dst = np.array([
        [0, 0],
        [width_mm, 0],
        [width_mm, height_mm],
        [0, height_mm]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(paper_corners.astype(np.float32), dst)
    return M, width_mm, height_mm

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def preprocess_structure_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(binary, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def detect_structure_and_load(image, scale_matrix):
    warped = cv2.warpPerspective(image, scale_matrix, (1000, 1400))
    edges = preprocess_structure_region(warped)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawing = warped.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # 小さいノイズを除外
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(drawing, [approx], -1, (0, 0, 255), 2)
    return drawing

# Streamlit UI
st.title("構造図スケール＆構造体・荷重検出")
paper_type = st.selectbox("用紙サイズを選んでください：", list(PAPER_SIZES.keys()))
uploaded_file = st.file_uploader("画像をアップロード：", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    corners = detect_paper_corners(image)
    if corners is None:
        st.error("用紙の輪郭が見つかりませんでした。明るい場所で用紙全体が写るように再撮影してください。")
    else:
        M, width_mm, height_mm = compute_scale(corners, PAPER_SIZES[paper_type])
        result = detect_structure_and_load(image, M)
        st.image(result, channels="BGR", caption="検出結果")
