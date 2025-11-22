import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image

st.title("構造図画像から応力図を自動生成（縮尺保持版）")

uploaded_file = st.file_uploader("構造図の画像をアップロード（縮尺が保たれるよう撮影推奨）", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 一時ファイルとして保存（PIL経由で読み込み）
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # OpenCV用に変換（BGR）
    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 処理前の表示（元の縮尺）
    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="元画像", use_column_width=False)

    # 処理スタート（縮尺変更なし）
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

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

    if len(support_x) >= 2 and load_x:
        left_support = min(support_x)
        right_support = max(support_x)
        load_position = int(np.mean(load_x))  # 複数検出された場合の平均

        L = right_support - left_support
        a = load_position - left_support
        b = L - a
        P = 10  # 荷重（仮）

        RA = P * b / L
        RB = P * a / L

        st.markdown(f"""
        ### 📐 応力計算結果  
        - 支点間距離 L: {L} px  
        - 荷重位置 a: {a} px, b: {b} px  
        - 左支点反力 RA = {RA:.2f} N  
        - 右支点反力 RB = {RB:.2f} N
        """)

    else:
        st.warning("支点や荷重が正しく検出できませんでした。もう一度鮮明な画像をアップロードしてみてください。")

    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="検出結果", use_column_width=False)
