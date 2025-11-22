import streamlit as st
import numpy as np
import cv2

st.title("手書き構造図の黒画素検出版検出過程表示アプリ")

uploaded_file = st.file_uploader("構造図画像をアップロードしてね", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (800, 600))

    st.subheader("元画像")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # グレースケール化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 黒画素検出（閾値を100に設定、100以下を黒＝255に、それ以外は白=0に反転）
    # ここでcv2.thresholdのTHRESH_BINARY_INV使うイメージ
    thresh_val = 100
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    st.subheader("黒画素検出（二値化画像）")
    st.image(thresh, use_column_width=True)

    # 輪郭抽出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭を元画像コピーに描画
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 255), 2)  # 黄色で輪郭表示

    st.subheader("輪郭検出結果（元画像に重ねて表示）")
    st.image(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB), use_column_width=True)

    # 白背景キャンバス作成（検出結果描画用）
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 255

    # 色・太さ定義
    beam_color = (255, 0, 0)    # 青
    support_color = (0, 0, 255) # 赤
    load_color = (0, 255, 0)    # 緑
    beam_thickness = 4
    triangle_height = 30

    detected_info = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            # 支点（三角形）描画
            pts = approx.reshape((-1, 2))
            cv2.drawContours(canvas, [pts], 0, support_color, -1)
            detected_info.append(f"支点（三角形） at x={x+w//2}, y={y+h//2}")

        elif h > w * 1.5:
            # 荷重（矢印）描画
            arrow_x = x + w//2
            arrow_y_start = y
            arrow_y_end = y + h

            cv2.line(canvas, (arrow_x, arrow_y_start), (arrow_x, arrow_y_end - 10), load_color, 4)

            arrow_tip = (arrow_x, arrow_y_end)
            arrow_left = (arrow_x - 10, arrow_y_end - 30)
            arrow_right = (arrow_x + 10, arrow_y_end - 30)
            arrow_head = np.array([arrow_tip, arrow_left, arrow_right])
            cv2.drawContours(canvas, [arrow_head], 0, load_color, -1)

            detected_info.append(f"荷重（矢印） at x={arrow_x}, y={arrow_y_end}")

        elif w > h * 3:
            # 梁（横長）
            beam_y = y + h//2
            cv2.line(canvas, (x, beam_y), (x + w, beam_y), beam_color, beam_thickness)
            detected_info.append(f"梁 at x={x}, y={beam_y}")

    st.subheader("検出結果（白背景に描画）")
    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_column_width=True)

    if detected_info:
        st.subheader("検出内容")
        for info in detected_info:
            st.write("- " + info)
    else:
        st.write("何も検出されませんでした。")

    # ダウンロード用関数
    def get_image_download_button(img_np, filename, label):
        _, buf = cv2.imencode('.png', img_np)
        st.download_button(
            label=label,
            data=buf.tobytes(),
            file_name=filename,
            mime="image/png"
        )

    get_image_download_button(img, "original_image.png", "元画像をダウンロード")
    get_image_download_button(thresh, "black_pixel_binary.png", "黒画素検出画像をダウンロード")
    get_image_download_button(img_contours, "contours_image.png", "輪郭検出画像をダウンロード")
    get_image_download_button(canvas, "detected_diagram.png", "検出図をダウンロード")
