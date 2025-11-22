import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("白黒二極化過程の可視化デモ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 画像読み込み
    pil_img = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_img)
    st.subheader("元画像")
    st.image(image, channels="RGB")

    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    st.subheader("グレースケール画像")
    st.image(gray, clamp=True, channels="GRAY")

    # 最も黒い画素と最も白い画素の値を検出
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    st.write(f"最も黒い画素値: {min_val}")
    st.write(f"最も白い画素値: {max_val}")

    # 黒・白極値の画素座標の可視化画像
    vis_points = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    min_coords = np.column_stack(np.where(gray == min_val))
    max_coords = np.column_stack(np.where(gray == max_val))

    vis_points[min_coords[:,0], min_coords[:,1]] = [255,0,0]   # 最黒点を赤
    vis_points[max_coords[:,0], max_coords[:,1]] = [0,255,0]   # 最白点を緑

    st.subheader("最も黒い点（赤）と最も白い点（緑）の位置")
    st.image(vis_points, channels="RGB")

    # ヒストグラム可視化
    hist_vals, bins = np.histogram(gray.flatten(), bins=256, range=[0,256])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(bins[:-1], hist_vals, color='black')
    ax.axvline(min_val, color='red', linestyle=':', label=f"min={min_val}")
    ax.axvline(max_val, color='green', linestyle=':', label=f"max={max_val}")
    ax.set_title("画素値ヒストグラム")
    ax.set_xlabel("画素値 (0=黒, 255=白)")
    ax.set_ylabel("画素数")
    ax.legend()
    st.pyplot(fig)

    # 二値化（min_valとmax_valでしきい値を自動決定）
    thresh = (min_val + max_val) // 2
    st.write(f"二値化しきい値: {thresh}")

    binary = (gray > thresh).astype(np.uint8) * 255
    st.subheader("二値化画像（白黒二極化）")
    st.image(binary, clamp=True, channels="GRAY")

    # すべての過程を並べて表示
    st.subheader("全過程まとめ")
    st.image([
        image,
        gray,
        vis_points,
        binary
    ], caption=[
        "元画像",
        "グレースケール",
        "最黒(赤)・最白(緑)可視化",
        "二値化"
    ], width=240)