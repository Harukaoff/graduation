import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.title("黒画素検出 vs エッジ検出 比較ツール")

uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    st.subheader("① 元画像")
    st.image(image_np, caption="元画像", channels="RGB")

    # --------------------------
    # 黒画素の検出（2値化＋黒密度マップ）
    # --------------------------
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    mask = (binary == 255).astype(np.uint8)
    density_map = cv2.blur(mask.astype(np.float32), (25, 25))

    st.subheader("② 黒画素密度マップ（ヒートマップ）")
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(density_map, cmap="hot", interpolation="bilinear")
    plt.colorbar(im, ax=ax1, label="黒画素の密度")
    st.pyplot(fig1)

    # --------------------------
    # エッジ検出（Canny）
    # --------------------------
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    st.subheader("③ エッジ検出（Canny法）")
    st.image(edges, caption="エッジ検出（黒＝エッジ）", channels="GRAY")

    # --------------------------
    # 並べて比較
    # --------------------------
    st.subheader("④ 並べて比較（わかりやすく）")
    col1, col2 = st.columns(2)
    with col1:
        st.text("黒画素密度（ヒートマップ）")
        st.pyplot(fig1)
    with col2:
        st.text("エッジ検出（Canny）")
        st.image(edges, caption="エッジ", channels="GRAY")
