import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

def compute_profile(binary_img, axis=0):
    return np.sum(binary_img == 255, axis=axis)

def find_peaks(profile, threshold=100):
    return np.where(profile > threshold)[0]

def merge_nearby(peaks, gap=5):
    if len(peaks) == 0:
        return []
    merged = []
    start = peaks[0]
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] > gap:
            merged.append((start, peaks[i - 1]))
            start = peaks[i]
    merged.append((start, peaks[-1]))
    return merged

def main():
    st.title("梁の検出アプリ（縦横のプロファイル + 黒部分可視化）")

    uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # 二値化（白＝背景、黒＝線）
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        st.subheader("① 黒検出（二値化画像）")
        st.image(binary, caption="黒（構造線）だけを抽出", channels="GRAY")

        # 横（Y方向）プロファイル
        y_profile = compute_profile(binary, axis=1)
        y_peaks = find_peaks(y_profile, threshold=100)
        y_lines = merge_nearby(y_peaks, gap=5)

        st.subheader("② 横方向の黒画素プロファイル（Y軸）")
        st.line_chart(y_profile)

        # 縦（X方向）プロファイル
        x_profile = compute_profile(binary, axis=0)
        x_peaks = find_peaks(x_profile, threshold=100)
        x_lines = merge_nearby(x_peaks, gap=5)

        st.subheader("③ 縦方向の黒画素プロファイル（X軸）")
        st.line_chart(x_profile)

        # 検出結果を描画（赤＝横線, 青＝縦線）
        result = image_np.copy()
        h, w = result.shape[:2]
        for y_start, y_end in y_lines:
            y = (y_start + y_end) // 2
            cv2.line(result, (0, y), (w - 1, y), (255, 0, 0), 2)
        for x_start, x_end in x_lines:
            x = (x_start + x_end) // 2
            cv2.line(result, (x, 0), (x, h - 1), (0, 0, 255), 2)

        st.subheader("④ 検出された梁ライン（赤：横、青：縦）")
        st.image(result, caption="検出された線分", channels="RGB")

if __name__ == "__main__":
    main()
