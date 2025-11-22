import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("構造図画像 編集ツール")

# アップロード
uploaded_file = st.file_uploader("構造図の画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    height, width, _ = image_np.shape

    st.image(image, caption="元画像", use_column_width=True)

    st.subheader("編集設定")

    # 各要素の位置をスライダーで編集（例として最大5個ずつ）
    num_beams = st.slider("梁の数", 1, 5, 1)
    beam_positions = []
    for i in range(num_beams):
        x1 = st.slider(f"梁{i+1} 始点x", 0, width, int(width * 0.2))
        y1 = st.slider(f"梁{i+1} 始点y", 0, height, int(height * 0.5))
        x2 = st.slider(f"梁{i+1} 終点x", 0, width, int(width * 0.8))
        y2 = st.slider(f"梁{i+1} 終点y", 0, height, int(height * 0.5))
        beam_positions.append(((x1, y1), (x2, y2)))

    num_supports = st.slider("支点の数", 1, 5, 1)
    support_positions = []
    for i in range(num_supports):
        x = st.slider(f"支点{i+1} x", 0, width, int(width * 0.2))
        y = st.slider(f"支点{i+1} y", 0, height, int(height * 0.5))
        support_positions.append((x, y))

    num_loads = st.slider("荷重の数", 1, 5, 1)
    load_positions = []
    for i in range(num_loads):
        x = st.slider(f"荷重{i+1} x", 0, width, int(width * 0.5))
        y = st.slider(f"荷重{i+1} y", 0, height, int(height * 0.3))
        load_positions.append((x, y))

    # 編集画像の描画
    annotated_image = image_np.copy()
    for start, end in beam_positions:
        cv2.line(annotated_image, start, end, (255, 0, 0), 3)

    for pos in support_positions:
        cv2.circle(annotated_image, pos, 10, (0, 255, 0), -1)

    for pos in load_positions:
        cv2.arrowedLine(annotated_image, (pos[0], pos[1]), (pos[0], pos[1]+40), (0, 0, 255), 3)

    st.subheader("編集後の画像")
    st.image(annotated_image, use_column_width=True)

    # ダウンロード用画像変換
    result_img = Image.fromarray(annotated_image)
    img_bytes = result_img.convert("RGB").tobytes("jpeg", "RGB")
    st.download_button("画像をダウンロード", data=img_bytes, file_name="edited_image.jpg", mime="image/jpeg")
