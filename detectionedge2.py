import streamlit as st
from streamlit_cropper import st_cropper
import cv2
import numpy as np
from PIL import Image
import os

# ヘッダー
st.title("構造図働線認識 WebApp")

# 画像アップロード
uploaded_file = st.file_uploader("構造図の画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロード画像", use_column_width=True)

    # トリミング範囲
    st.subheader("切り取り範囲を選択")
    cropped_img = st_cropper(image, realtime_update=True, box_color='#0000FF', aspect_ratio=None)

    # トリミング後の処理
    opencv_img = np.array(cropped_img)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)

    # テンプレートの読み込み
    template_paths = {
        'fixed': 'templates/fixed.png',
        'pin': 'templates/pin.png',
        'roller': 'templates/roller.png',
        'force': 'templates/force.png'
    }

    templates = {}
    for name, path in template_paths.items():
        temp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if temp is None:
            st.error(f"テンプレート {name}.png が見つかりません: {path}")
            continue
        templates[name] = temp

    # テンプレート統合検出
    gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
    display_img = opencv_img.copy()

    colors = {
        'fixed': (255, 0, 0),       # 赤
        'pin': (0, 255, 0),         # 緑
        'roller': (0, 0, 255),      # 青
        'force': (0, 255, 255)      # アクア
    }

    for name, temp in templates.items():
        w, h = temp.shape[::-1]
        res = cv2.matchTemplate(gray, temp, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(display_img, pt, (pt[0] + w, pt[1] + h), colors[name], 2)
            cv2.putText(display_img, name, (pt[0], pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1)

    # 結果表示
    st.subheader("認識結果")
    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="認識結果", use_column_width=True)

# デバッグ用
# st.write("現在のカレントディレクトリ:", os.getcwd())
