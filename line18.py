import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🧠 構造図から梁と支点（三角形）を検出するアプリ")

uploaded_file = st.file_uploader("構造図画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # グレースケール＆二値化
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 膨張処理で線をつなぐ
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # 輪郭検出
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # DEBUG: 全輪郭を表示
    debug_img = image_np.copy()
    cv2.drawContours(debug_img, contours, -1, (0, 255, 255), 1)
    st.image(debug_img, caption="🔍 DEBUG: 全輪郭（黄色）", channels="RGB")

    result_img = image_np.copy()
    beam_found = False
    support_found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue

        # 直線（梁）候補
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h != 0 else 0
        if aspect_ratio > 5 and w > 100:
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            beam_found = True
            continue

        # 三角形（支点）候補
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            cv2.drawContours(result_img, [cnt], 0, (0, 0, 255), 2)
            support_found = True

    st.image(result_img, caption="🔧 検出結果（梁=緑、支点=赤）", channels="RGB")

    # 結果メッセージ
    if beam_found:
        st.success("✅ 梁（長い直線）が検出されました")
    else:
        st.warning("🟡 梁（長い直線）が検出されませんでした")

    if support_found:
        st.success("✅ 支点（三角形）が検出されました")
    else:
        st.warning("🟡 支点（三角形）が検出されませんでした")

