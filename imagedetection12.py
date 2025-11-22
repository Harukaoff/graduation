import streamlit as st
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_np):
    # グレースケール変換
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # ヒストグラム均等化で明るさ補正
    gray = cv2.equalizeHist(gray)
    # ノイズ除去
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 二値化（大津の方法で自動しきい値）
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def detect_components(image_np, thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    support_x = []
    load_x = []
    beam_coords = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            # 支点（三角形）
            cv2.drawContours(image_np, [approx], 0, (0, 255, 0), 2)
            support_x.append(x + w // 2)
        elif h > w * 1.5:
            # 荷重（縦長）
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
            load_x.append(x + w // 2)
        elif w > h * 3:
            # 梁（横長）
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 0, 255), 2)
            beam_coords.append((x, y, w, h))

    return image_np, support_x, load_x, beam_coords

def calculate_reactions(support_x, load_x, load_value=10):
    if len(support_x) >= 2 and load_x:
        left_support = min(support_x)
        right_support = max(support_x)
        load_position = int(np.mean(load_x))

        L = right_support - left_support
        a = load_position - left_support
        b = L - a

        RA = load_value * b / L
        RB = load_value * a / L

        return True, RA, RB, L, a, b
    else:
        return False, 0, 0, 0, 0, 0

# Streamlit UI
st.title("構造図画像から応力図を高精度で自動生成")

uploaded_file = st.file_uploader("構造図画像をアップロード（縮尺そのまま・鮮明な画像推奨）", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="元画像", use_column_width=False)

    thresh = preprocess_image(image_np)
    processed_image, support_x, load_x, beam_coords = detect_components(image_np.copy(), thresh)

    detected, RA, RB, L, a, b = calculate_reactions(support_x, load_x)

    if detected:
        st.markdown(f"""
        ### 📐 応力計算結果  
        - 支点間距離 L: {L} px  
        - 荷重位置 a: {a} px, b: {b} px  
        - 左支点反力 RA = {RA:.2f} N  
        - 右支点反力 RB = {RB:.2f} N
        """)
    else:
        st.warning("構造部材が正しく認識できませんでした。明るく鮮明な画像で再度試してみてください。")

    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="検出結果（強化処理済み）", use_column_width=False)
