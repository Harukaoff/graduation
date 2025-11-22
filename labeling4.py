import streamlit as st
import cv2
import numpy as np
import os

st.title("構造図の要素自動ラベリングアプリ")

# === テンプレ画像の読み込み ===
TEMPLATE_DIR = r"C:\Users\morim\Downloads\graduation\templates"

template_files = {
    "ピン支点": "pin2.png",
    "ローラー支点": "roller2.png",
    "固定支点": "fixed1.png",
    "ヒンジ": "hinge.png",
    "荷重": "kajyu.png",
    "モーメント荷重": "moment.jpeg",
}

template_contours = {}
for label, filename in template_files.items():
    path = os.path.join(TEMPLATE_DIR, filename).replace("\\", "/")
    template_img = cv2.imread(path, 0)

    if template_img is None:
        st.error(f"❌ テンプレ画像の読み込みに失敗: {path}")
        continue

    _, thresh = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        template_contours[label] = contours[0]
    else:
        st.warning(f"{label} の輪郭が検出できませんでした")

# === ユーザー画像のアップロード ===
uploaded_file = st.file_uploader("構造図画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = input_img.copy()

    for cnt in contours:
        best_label = None
        best_score = float("inf")

        for label, tmpl_cnt in template_contours.items():
            score = cv2.matchShapes(cnt, tmpl_cnt, 1, 0.0)
            if score < best_score:
                best_score = score
                best_label = label

        if best_score < 0.1:
            x, y, w, h = cv2.boundingRect(cnt)
            roi_thresh = thresh[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]

            # 輪郭マスクと白ピクセル判定（白抜き対応）
            mask = np.zeros_like(roi_thresh)
            cv2.drawContours(mask, [cnt - [x, y]], -1, 255, -1)

            inside_pixels = cv2.countNonZero(cv2.bitwise_and(roi_thresh, roi_thresh, mask=mask))
            area = cv2.contourArea(cnt)
            white_ratio = inside_pixels / area if area > 0 else 0
            is_hollow = white_ratio > 0.5

            # Hough変換で下部の水平線検出（ローラー支点用）
            lines = cv2.HoughLinesP(roi_thresh, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=5)
            has_bottom_line = False
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(y1 - y2) < 3 and min(y1, y2) > h * 0.6:
                        has_bottom_line = True
                        break

            # 白抜き三角形の支点分類を強制上書き
            if is_hollow:
                best_label = "ローラー支点" if has_bottom_line else "ピン支点"

            # ラベリング表示
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_img, best_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    st.image(result_img, channels="BGR", caption="ラベル付け結果", use_column_width=True)
