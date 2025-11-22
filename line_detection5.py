import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("構造図アップロード＆反力計算アプリ")

# 画像アップロード
uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 画像表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた構造図", use_column_width=True)

    # 仮データ（ここは画像解析で自動取得するように将来アップグレードできる）
    beam_data = {
        "beam_length": 10.0,
        "supports": [
            {"type": "pin", "position": 0.0},
            {"type": "roller", "position": 10.0}
        ],
        "loads": [
            {"type": "point", "position": 4.0, "magnitude": 10.0}
        ]
    }

    def calculate_reactions(beam_length, supports, loads):
        A = supports[0]['position']
        B = supports[1]['position']
        L = B - A

        RA = 0
        RB = 0

        for load in loads:
            if load['type'] == 'point':
                P = load['magnitude']
                x = load['position'] - A
                RB += (P * x) / L

        total_load = sum([load['magnitude'] for load in loads])
        RA = total_load - RB

        return RA, RB

    # 反力計算
    RA, RB = calculate_reactions(
        beam_length=beam_data["beam_length"],
        supports=beam_data["supports"],
        loads=beam_data["loads"]
    )

    st.subheader("反力の計算結果")
    st.write(f"左端支点反力 RA = {RA:.2f} N")
    st.write(f"右端支点反力 RB = {RB:.2f} N")
