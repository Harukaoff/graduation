import cv2
import numpy as np
import streamlit as st

st.title("テンプレートマッチングによる構造図ラベリング")

# テンプレートの読み込み
template_files = {
    "ピン": "templates/pin2.png",
    "ローラー": "templates/roller2.png",
    "固定": "templates/fixed1.png",
    "ヒンジ": "templates/hinge.png",
    "荷重": "templates/kajyu.png"
}
templates = {label: cv2.imread(path, 0) for label, path in template_files.items()}

# 入力画像アップロード
uploaded = st.file_uploader("構造図をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded:
    buf = np.frombuffer(uploaded.read(), dtype=np.uint8)
    src = cv2.imdecode(buf, 1)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    output = src.copy()

    detections = []

    for label, templ in templates.items():
        w, h = templ.shape[::-1]
        res = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.6)  # 類似度しきい値

        for pt in zip(*loc[::-1]):
            # 重複除去（近傍に同じラベルがある場合スキップ）
            if any(abs(pt[0]-x)<10 and abs(pt[1]-y)<10 for (_, x, y) in detections):
                continue
            detections.append((label, pt[0], pt[1]))
            cv2.rectangle(output, pt, (pt[0]+w, pt[1]+h), (0, 255, 0), 2)
            cv2.putText(output, label, (pt[0], pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption=f"検出数: {len(detections)}個", use_column_width=True)

    if len(detections) == 0:
        st.warning("テンプレートに一致する要素は見つかりませんでした。テンプレートサイズや画像品質を確認してください。")
