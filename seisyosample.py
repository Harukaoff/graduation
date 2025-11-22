import streamlit as st
import cv2
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

# YOLO モデルをロード
model = YOLO("runs/obb/train7/weights/best.pt")  # 学習済みモデルのパスに変更

st.title("構造図清書アプリ")

# アップロード
uploaded_file = st.file_uploader("構造図の画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 一時保存
    input_path = os.path.join("temp_input.png")
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    # YOLOで推論
    results = model.predict(source=input_path, save=False, conf=0.5)

    # 元画像読み込み
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 清書処理（シンプルに検出枠を描画）
    for box in results[0].obb.xyxy:  # OBBの四隅座標を取得
        pts = box.cpu().numpy().astype(int).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # 結果を保存
    output_path = "output_cleaned.png"
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    st.image(img, caption="清書した構造図", use_container_width=True)
    st.success(f"清書結果を {output_path} に保存しました ✅")

