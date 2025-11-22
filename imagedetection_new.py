import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("構造図アップローダーと反力計算")

uploaded_file = st.file_uploader("構造図の画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # PIL→OpenCV形式に変換
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (800, 600))

    # グレースケール & 二値化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    # 輪郭検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    left_support_x = None
    right_support_x = None
    load_x = None
    load_value = st.number_input("荷重の大きさ（N）", min_value=0.0, value=10.0)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        # 支点（三角形）
        if len(approx) == 3:
            cx = x + w // 2
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
            st.write("支点（三角形）検出:", cx)

            if left_support_x is None:
                left_support_x = cx
            else:
                right_support_x = cx if cx > left_support_x else left_support_x
                left_support_x = min(left_support_x, cx)

        # 荷重（矢印）
        elif h > w * 1.5:
            cx = x + w // 2
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            st.write("荷重（縦長）検出:", cx)
            load_x = cx

        # 梁（横長）
        elif w > h * 3:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            st.write("梁（横長）検出:", x, y)

    # 反力計算
    if left_support_x is not None and right_support_x is not None and load_x is not None:
        L = abs(right_support_x - left_support_x)
        a = abs(load_x - left_support_x)
        b = L - a

        RA = load_value * b / L
        RB = load_value * a / L

        st.success(f"左端支点反力 RA = {RA:.2f} N")
        st.success(f"右端支点反力 RB = {RB:.2f} N")
    else:
        st.warning("支点または荷重が検出できなかったため、反力は計算できません。")

    # BGR→RGBで表示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="解析結果", use_column_width=True)
