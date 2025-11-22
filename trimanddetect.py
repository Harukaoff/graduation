import cv2
import numpy as np
import streamlit as st

st.title("支点判別＋選択範囲表示アプリ")

uploaded_file = st.file_uploader("構造図画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    height, width = image.shape[:2]

    # 範囲スライダー（0〜100%）
    col1, col2 = st.columns(2)
    with col1:
        left_perc = st.slider("左端 (0〜100%)", 0, 99, 0)
    with col2:
        right_perc = st.slider("右端 (0〜100%)", left_perc + 1, 100, 100)

    # 実際のピクセル位置に変換
    x1 = int(width * (left_perc / 100))
    x2 = int(width * (right_perc / 100))

    # 青い縦線で範囲表示
    image_with_lines = image.copy()
    cv2.line(image_with_lines, (x1, 0), (x1, height), (255, 0, 0), 2)
    cv2.line(image_with_lines, (x2, 0), (x2, height), (255, 0, 0), 2)

    # ROI内だけを対象に処理
    roi = image[:, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Hough変換で線検出
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=5)
    horizontal_lines = []

    if lines is not None:
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            angle = np.arctan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi
            if -10 < angle < 10:  # 水平線
                horizontal_lines.append(((x1_l, y1_l), (x2_l, y2_l)))
                cv2.line(roi, (x1_l, y1_l), (x2_l, y2_l), (0, 0, 255), 2)

    # 三角形検出（輪郭から）
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3 and cv2.contourArea(cnt) > 100:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.drawContours(roi, [approx], 0, (0, 255, 255), 2)
                cv2.circle(roi, (cx, cy), 4, (0, 255, 255), -1)

                is_roller = False
                for line in horizontal_lines:
                    x1_l, y1_l = line[0]
                    x2_l, y2_l = line[1]
                    y_avg = (y1_l + y2_l) // 2
                    if abs(y_avg - cy) < 20 and abs(cx - ((x1_l + x2_l) // 2)) < 30:
                        is_roller = True
                        cv2.putText(roi, "Roller", (cx - 30, cy + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        break
                if not is_roller:
                    cv2.putText(roi, "Pin", (cx - 20, cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 画像表示
    st.image(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB), caption="選択範囲と元画像", use_column_width=True)
    st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption="選択範囲内の支点判別", use_column_width=True)
