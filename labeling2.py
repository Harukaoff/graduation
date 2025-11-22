import cv2
import numpy as np
import streamlit as st

# テンプレートフォルダ
TEMPLATE_DIR = "C:/Users/morim/Documents/graduation/templates/"

# テンプレート画像の読み込み（グレースケール）
templates = {
    "pin": cv2.imread(TEMPLATE_DIR + "pin2.png", 0),
    "roller": cv2.imread(TEMPLATE_DIR + "roller2.png", 0),
    "fixed": cv2.imread(TEMPLATE_DIR + "fixed1.png", 0),
    "hinge": cv2.imread(TEMPLATE_DIR + "hinge.png", 0),
    "load": cv2.imread(TEMPLATE_DIR + "kajyu.png", 0),
    "moment": cv2.imread(TEMPLATE_DIR + "moment.jpeg", 0),
}

# テンプレートごとの輪郭抽出
template_contours = {}
for key, img in templates.items():
    if img is None:
        st.error(f"{key} のテンプレート画像が読み込めませんでした。パスを確認してください。")
        continue
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        template_contours[key] = contours[0]  # 最大の輪郭だけ使用

# 検出関数：画像内の全輪郭とテンプレートの輪郭を比較
def detect_elements(image_bytes, templates_dict):
    img = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < 100:  # 小さいノイズ除去
            continue

        match_scores = {}
        for name, tmpl_cnt in templates_dict.items():
            score = cv2.matchShapes(cnt, tmpl_cnt, 1, 0.0)
            match_scores[name] = score

        best_match = min(match_scores, key=match_scores.get)
        if match_scores[best_match] < 0.3:  # 形が似ているかどうかの閾値
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, best_match, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img

# Streamlit UI
st.title("構造図要素の自動検出")

uploaded_file = st.file_uploader("構造図画像をアップロード（png or jpg）", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    result_img = detect_elements(uploaded_file, template_contours)
    st.image(result_img, caption="検出結果", channels="BGR")