import streamlit as st
import cv2
import numpy as np
import os

st.title("構造図の要素自動ラベリングアプリ")

# === テンプレ画像の読み込み ===
TEMPLATE_DIR =TEMPLATE_DIR = r"C:\Users\morim\Downloads\graduation\templates"

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

        if best_score < 0.1:  # しきい値（適宜調整）
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_img, best_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    st.image(result_img, channels="BGR", caption="ラベル付け結果", use_column_width=True)
