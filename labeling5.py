import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("構造図の要素検出・ラベリング")

uploaded = st.file_uploader("構造図画像をアップロード", type=["jpg","png","jpeg"])
if not uploaded:
    st.info("画像をアップロードしてください。")
    st.stop()

# 画像読み込み＆前処理
file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
orig = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binarized = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# 輪郭抽出
contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 200: 
        continue
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    label = None
    color = (0, 0, 255)
    if len(approx) == 3:
        # 三角 → ローラーかピン
        # 下方向に水平線があるか確認
        x, y, w, h = cv2.boundingRect(cnt)
        roi = binarized[y+h:y+h+10, x:x+w]
        lines = cv2.HoughLinesP(roi,1,np.pi/180,threshold=20,minLineLength=w//2,maxLineGap=3)
        if lines is not None:
            label = "Roller"
            color = (0, 0, 255)
        else:
            label = "Pin"
            color = (255, 0, 0)
    elif len(approx) == 4:
        label = "Fixed"
        color = (0, 128, 0)
    elif len(approx) > 8:
        # 円ぽいもの → ヒンジかモーメント荷重
        if area < 5000:
            label = "Hinge"
            color = (128, 0, 128)
        else:
            label = "Moment"
            color = (128, 128, 0)
    # 最後に矢印検知（荷重）
    # 長く尖った頂点があるかチェック
    if len(approx) == 7 and label is None:
        label = "Load"
        color = (0, 255, 255)

    if label:
        cv2.drawContours(orig, [cnt], -1, color, 2)
        cv2.putText(orig, label, (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="検出結果", use_column_width=True)
