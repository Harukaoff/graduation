import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

st.title("黒画素 vs エッジ（Canny）比較 + 閾値可視化")

uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    st.subheader("① 元画像")
    st.image(image_np, caption="元画像", channels="RGB")

    # ==========================
    # 黒画素ヒートマップ
    # ==========================
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    mask = (binary == 255).astype(np.uint8)
    density_map = cv2.blur(mask.astype(np.float32), (25, 25))

    st.subheader("② 黒画素密度マップ")
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(density_map, cmap="hot", interpolation="bilinear")
    plt.colorbar(im, ax=ax1, label="黒密度")
    st.pyplot(fig1)

    # ==========================
    # Cannyしきい値スライダー
    # ==========================
    st.subheader("③ Cannyエッジ検出（しきい値調整）")

    t1 = st.slider("下限 threshold1", 0, 255, 50)
    t2 = st.slider("上限 threshold2", 0, 255, 150)

    edges = cv2.Canny(gray, threshold1=t1, threshold2=t2)
    st.image(edges, caption=f"Cannyエッジ（t1={t1}, t2={t2}）", channels="GRAY")

    # ==========================
    # ④ 閾値の色見本バー
    # ==========================
    st.subheader("④ 閾値がどの明るさか（グレースケール対応）")

    bar = np.linspace(0, 255, 256).astype(np.uint8)
    grad = np.tile(bar, (50, 1))  # 高さ50pxのバー

    fig2, ax2 = plt.subplots(figsize=(8, 2))
    ax2.imshow(grad, cmap="gray", aspect="auto")
    ax2.set_title("画素値のグレースケール対応（0=黒, 255=白）")

    # 縦線で現在のt1, t2を表示
    ax2.axvline(t1, color="blue", label=f"t1={t1}")
    ax2.axvline(t2, color="red", label=f"t2={t2}")
    ax2.legend()
    ax2.set_xticks([0, 64, 128, 192, 255])
    ax2.set_yticks([])

    st.pyplot(fig2)
