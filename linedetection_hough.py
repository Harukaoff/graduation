import streamlit as st
import numpy as np
import cv2

st.title("ハフ変換＋輪郭で 梁・支点・荷重 検出アプリ")

uploaded_file = st.file_uploader("手書き構造図画像をアップロードしてね", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (800, 600))

    st.subheader("元画像")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 黒画素抽出（二値化）
    thresh_val = 100
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    st.subheader("二値化画像（黒画素検出）")
    st.image(binary, use_column_width=True)

    # ハフ変換用にエッジ検出（Canny）
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    st.subheader("エッジ検出画像 (Canny)")
    st.image(edges, use_column_width=True)

    # ハフ変換で直線検出
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # 白背景キャンバス作成
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # 直線（梁）を青で描画
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            # 横長直線のフィルター（高さ差が小さいもの）
            if abs(y2 - y1) < 10 and length > 100:
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 4)

    # 輪郭検出（三角形支点・縦長荷重検出用）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 色定義
    support_color = (0, 0, 255)  # 赤（三角形支点）
    load_color = (0, 255, 0)     # 緑（荷重）
    detected_info = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            # 三角形 → 支点
            pts = approx.reshape((-1, 2))
            cv2.drawContours(canvas, [pts], 0, support_color, -1)
            detected_info.append(f"支点（三角形） at x={x+w//2}, y={y+h//2}")

        elif h > w * 1.5 and w > 10 and h > 30:
            # 縦長で大きい → 荷重矢印
            arrow_x = x + w//2
            arrow_y_start = y
            arrow_y_end = y + h

            # 線部分
            cv2.line(canvas, (arrow_x, arrow_y_start), (arrow_x, arrow_y_end - 10), load_color, 4)

            # 鋭利矢印（三角形）
            arrow_tip = (arrow_x, arrow_y_end)
            arrow_left = (arrow_x - 10, arrow_y_end - 30)
            arrow_right = (arrow_x + 10, arrow_y_end - 30)
            arrow_head = np.array([arrow_tip, arrow_left, arrow_right])
            cv2.drawContours(canvas, [arrow_head], 0, load_color, -1)

            detected_info.append(f"荷重（矢印） at x={arrow_x}, y={arrow_y_end}")

    st.subheader("検出結果（白背景に描画）")
    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_column_width=True)

    if detected_info:
        st.subheader("検出内容")
        for info in detected_info:
            st.write("- " + info)
    else:
        st.write("支点・荷重は検出されませんでした。")

    # ダウンロードボタン用関数
    def get_image_download_button(img_np, filename, label):
        _, buf = cv2.imencode('.png', img_np)
        st.download_button(label=label, data=buf.tobytes(), file_name=filename, mime="image/png")

    get_image_download_button(img, "original_image.png", "元画像をダウンロード")
    get_image_download_button(binary, "binary_image.png", "二値化画像をダウンロード")
    get_image_download_button(edges, "edges_image.png", "エッジ検出画像をダウンロード")
    get_image_download_button(canvas, "detected_diagram.png", "検出図をダウンロード")
