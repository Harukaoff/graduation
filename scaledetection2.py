import streamlit as st
import cv2
import numpy as np
import imutils
from PIL import Image
import tempfile
import os

# -----------------------------
# 1. 用紙サイズ（mm）
# -----------------------------
PAPER_SIZES = {
    'A4': (210, 297),
    'B4': (257, 364),
    'B5': (182, 257),
}

def get_paper_dimensions(paper_type):
    return PAPER_SIZES.get(paper_type.upper(), PAPER_SIZES['A4'])

# -----------------------------
# 2. スケール計算（pixels/mm）
# -----------------------------
def get_pixels_per_mm(image, paper_type='A4'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if not contours:
        raise Exception("紙の輪郭が見つかりません")

    paper_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(paper_contour)
    width, height = sorted(rect[1])  # pixels

    real_w, real_h = get_paper_dimensions(paper_type)
    scale = max(width / real_w, height / real_h)
    return scale, paper_contour

# -----------------------------
# 3. 構造要素検出
# -----------------------------
def detect_structure_elements(image, scale):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=80,
                            minLineLength=int(50 * scale), maxLineGap=int(5 * scale))
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    contours = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 3 and area > 50:
            cv2.drawContours(output, [approx], -1, (255, 0, 0), 2)

    return output

# -----------------------------
# 4. Streamlit UI
# -----------------------------
def main():
    st.title("📐 構造図自動認識ツール（用紙サイズ選択付き）")
    st.write("用紙サイズと構造図画像をアップロードしてください。")

    paper_type = st.selectbox("用紙サイズを選択", ["A4", "B4", "B5"])

    uploaded_file = st.file_uploader("構造図画像をアップロード（紙の上に描かれたもの）", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image, caption="アップロードされた画像", use_column_width=True)

        try:
            scale, contour = get_pixels_per_mm(image, paper_type)
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

            result = detect_structure_elements(image, scale)

            st.success(f"検出成功！スケール: {scale:.2f} pixels/mm")
            st.image(result, caption="検出結果", use_column_width=True)

            # 保存してダウンロードリンクも作る
            result_path = os.path.join(tempfile.gettempdir(), "result.jpg")
            cv2.imwrite(result_path, result)
            with open(result_path, "rb") as f:
                st.download_button("結果画像をダウンロード", f, file_name="result.jpg", mime="image/jpeg")

        except Exception as e:
            st.error(f"エラー: {str(e)}")

if __name__ == "__main__":
    main()
