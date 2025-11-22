import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def compute_black_density_map(binary_img, ksize=25):
    # 黒画素（255）を 1、白を 0 とした強度マップを平滑化
    mask = (binary_img == 255).astype(np.uint8)
    density_map = cv2.blur(mask.astype(np.float32), (ksize, ksize))
    return density_map

def main():
    st.title("黒さの検出度マップ（コンター・ヒートマップ風表示）")

    uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        st.subheader("① 元画像")
        st.image(image_np, caption="元画像", channels="RGB")

        st.subheader("② 二値化画像（黒＝255）")
        st.image(binary, caption="二値画像", channels="GRAY")

        st.subheader("③ 黒さの検出度マップ（滑らかにした強度）")
        density_map = compute_black_density_map(binary, ksize=25)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.imshow(density_map, cmap="hot", interpolation="bilinear")
        plt.colorbar(label="黒画素の密度（0〜1）")
        plt.title("黒検出度のヒートマップ")
        st.pyplot(fig)

        st.subheader("④ 元画像とヒートマップの重ね合わせ（視覚的説明用）")
        heatmap_color = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        blend = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
        st.image(blend, caption="元画像 + 検出ヒートマップ", channels="RGB")

if __name__ == "__main__":
    main()
