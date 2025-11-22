import cv2
import numpy as np
from PIL import Image
import streamlit as st

def detect_triangles_and_beam(image_pil,
                              blur_kernel=5,
                              threshold_type="Adaptive Gaussian",
                              block_size=29,
                              C_value=2,
                              canny_lower=50,
                              canny_upper=150,
                              min_area=1720,
                              approx_epsilon_factor=0.2):
    # 画像前処理
    image_np = np.array(image_pil.convert('RGB'))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # 二値化
    if threshold_type == "Otsu":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_type == "Adaptive Mean":
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, C_value)
    elif threshold_type == "Adaptive Gaussian":
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, C_value)
    else:
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # エッジ検出
    edges = cv2.Canny(binary, canny_lower, canny_upper)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = image_np.copy()
    triangle_apexes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        epsilon = approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            pts = approx.reshape(-1, 2)
            # y座標が最小の頂点（画像上で最も上）を上端とみなす
            top_idx = np.argmin(pts[:, 1])
            apex = tuple(pts[top_idx])
            triangle_apexes.append(apex)
            cv2.drawContours(result_img, [approx], 0, (0, 255, 0), 3)
            cv2.circle(result_img, apex, 8, (255, 0, 0), -1)
            cv2.putText(result_img, "Apex", apex, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 三角形の上端同士を直線で結ぶ（梁）
    if len(triangle_apexes) >= 2:
        triangle_apexes = sorted(triangle_apexes, key=lambda x: x[0]) # x座標で左右の三角形を判定
        cv2.line(result_img, triangle_apexes[0], triangle_apexes[-1], (0, 0, 255), 6)
        # 途中の三角形も順に結びたい場合
        for i in range(len(triangle_apexes)-1):
            cv2.line(result_img, triangle_apexes[i], triangle_apexes[i+1], (0,0,255), 4)

    return result_img, triangle_apexes

st.title("三角形検出＋梁描画デモ（パラメータ指定版）")
uploaded_file = st.file_uploader("画像をアップロード", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    # パラメータは固定値で呼び出し（UIで調整したい場合はst.sidebar.sliderなどに変更）
    result_img, apexes = detect_triangles_and_beam(
        img,
        blur_kernel=5,
        threshold_type="Adaptive Gaussian",
        block_size=29,
        C_value=2,
        canny_lower=50,
        canny_upper=150,
        min_area=1720,
        approx_epsilon_factor=0.2
    )
    st.image(result_img, caption="三角形の上端（青）を赤線で結んだ梁", use_column_width=True)
    st.write("検出された三角形の上端座標:", apexes)