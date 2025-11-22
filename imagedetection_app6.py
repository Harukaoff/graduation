import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="構造図解析ツール", layout="centered")
st.title("🛠 構造図自動認識ツール（梁・支点・荷重）")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("画像の読み込みに失敗しました。")
        st.stop()

    display_img = img.copy()

    # ① グレースケール & ぼかし
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ② Cannyエッジ検出（直線用）
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # ③ ハフ変換：梁（横長な直線）
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
    beam_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < abs(x2 - x1):  # 横向き
                cv2.line(display_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                beam_count += 1

    # ④ 二値化（支点と荷重検出用）
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    # ⑤ ノイズ除去（開演算）
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # ⑥ 輪郭抽出
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    support_count = 0
    load_count = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        # 支点（三角形）検出
        if len(approx) == 3:
            cv2.drawContours(display_img, [approx], 0, (0, 255, 0), 2)
            cv2.putText(display_img, "支点", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            support_count += 1

        # 荷重（縦長長方形）
        elif h > w * 1.5:
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(display_img, "荷重", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            load_count += 1

    # 結果表示
    st.subheader("検出結果")
    st.markdown(f"- 🔴 梁: **{beam_count}本**")
    st.markdown(f"- 🟢 支点: **{support_count}個**")
    st.markdown(f"- 🔵 荷重: **{load_count}個**")

    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="検出された構造図", use_column_width=True)
