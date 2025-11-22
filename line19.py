import streamlit as st
import cv2
import numpy as np

st.title("構造図 支点・荷重・梁 検出アプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ファイルをOpenCVで扱える形式に変換
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # リサイズ
    img = cv2.resize(img, (800, 600))

    # 処理用コピー
    img_draw = img.copy()

    # グレースケール＆二値化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    # 輪郭抽出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        # 支点（三角形）
        if len(approx) == 3:
            cv2.drawContours(img_draw, [approx], 0, (0, 255, 0), 2)
            results.append(f"Triangle (支点) detected at x={x + w // 2}, y={y + h // 2}")

        # 荷重（矢印） 縦長の図形
        elif h > w * 1.5:
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 0, 0), 2)
            results.append(f"Vertical shape (矢印？) at x={x + w // 2}, y={y + h // 2}")

        # 梁（直線） 横長なもの
        elif w > h * 3:
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 2)
            results.append(f"Beam (梁) at x={x}, y={y}")

    # RGBに変換してStreamlit表示用に
    img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="検出結果", use_column_width=True)

    # 検出結果を表示
    if results:
        for r in results:
            st.write(r)
    else:
        st.write("検出結果なし")

