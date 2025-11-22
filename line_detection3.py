import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("構造図画像から応力解析！")

uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)
    
    # --- 描画処理（例：固定された構造図の場合） ---
    # 画像サイズ取得
    img_height, img_width = image.shape[:2]

    # 梁（赤色）
    beam_y = img_height - 50
    cv2.line(image, (50, beam_y), (img_width - 50, beam_y), (0, 0, 255), 4)

    # 左支点（三角形・青）
    left_support_base = 50
    triangle_height = 30
    triangle_left = np.array([
        [left_support_base, beam_y],
        [left_support_base - 15, beam_y + triangle_height],
        [left_support_base + 15, beam_y + triangle_height]
    ])
    cv2.drawContours(image, [triangle_left], 0, (255, 0, 0), -1)

    # 右支点（三角形・青）
    right_support_base = img_width - 50
    triangle_right = np.array([
        [right_support_base, beam_y],
        [right_support_base - 15, beam_y + triangle_height],
        [right_support_base + 15, beam_y + triangle_height]
    ])
    cv2.drawContours(image, [triangle_right], 0, (255, 0, 0), -1)

    # 荷重（中央矢印・黄色）
    arrow_x = (left_support_base + right_support_base) // 2
    arrow_y_start = beam_y - 80
    arrow_y_end = beam_y - 4
    cv2.arrowedLine(image, (arrow_x, arrow_y_start), (arrow_x, arrow_y_end), (0, 255, 255), 4, tipLength=0.2)

    st.image(image, caption="色分けされた構造図", channels="BGR")

    # 仮データ（例として固定）
    beam_data = {
        "beam_length": 10.0,
        "supports": [
            {"type": "pin", "position": 0.0},
            {"type": "roller", "position": 10.0}
        ],
        "loads": [
            {"type": "point", "position": 5.0, "magnitude": 10.0}
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

    RA, RB = calculate_reactions(
        beam_length=beam_data["beam_length"],
        supports=beam_data["supports"],
        loads=beam_data["loads"]
    )

    st.markdown("### ⚙️ 反力計算結果")
    st.write(f"左端支点反力 RA = {RA:.2f} N")
    st.write(f"右端支点反力 RB = {RB:.2f} N")
