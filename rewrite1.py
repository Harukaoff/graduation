import streamlit as st
import cv2
import numpy as np

st.title("手書き構造図 → 清書図面（支点は三角形検出で）")

uploaded_file = st.file_uploader("手書きの構造図をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # ファイル読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    st.image(input_img, caption="アップロード画像", channels="BGR")

    # エッジ検出（支点検出用）
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    support_triangles = []
    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 3 and cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(approx)
            support_triangles.append((x, y, w, h))

    # 梁（長い直線）の検出
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
    beams = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 100:
                beams.append((x1, y1, x2, y2))

    # 白紙のキャンバスに清書
    h, w = input_img.shape[:2]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    # 梁を描画
    for (x1, y1, x2, y2) in beams:
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 0), thickness=4)

    # 支点（三角形）を描画
    for (x, y, w, h) in support_triangles:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(canvas, "支点", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 表示
    st.image(canvas, caption="清書された構造図", channels="BGR")

    # JSON形式で構造データ出力
    structure_data = {
        "beams": beams,
        "supports": [{"type": "triangle", "x": x, "y": y, "w": w, "h": h} for (x, y, w, h) in support_triangles]
    }
    st.json(structure_data)
