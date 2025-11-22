import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="構造図の特徴検出", layout="wide")
st.title("SIFT + ホモグラフィによるテンプレートマッチング（スケール＆回転対応）")

uploaded_img = st.sidebar.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])
uploaded_template = st.sidebar.file_uploader("テンプレート画像をアップロード", type=["png", "jpg", "jpeg"])
label_name = st.sidebar.text_input("ラベル名（任意）", "マッチ")

if uploaded_img and uploaded_template:
    img = Image.open(uploaded_img).convert("RGB")
    templ = Image.open(uploaded_template).convert("RGB")
    img_np = np.array(img)
    templ_np = np.array(templ)

    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    templ_gray = cv2.cvtColor(templ_np, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(templ_gray, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    result_img = img_np.copy()
    if len(good) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = templ_gray.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        result_img = cv2.polylines(result_img, [np.int32(dst)], True, (0, 255, 0), 3)

        # ラベル
        x, y = np.int32(dst[0][0])
        cv2.putText(result_img, label_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        st.image(result_img, caption="検出結果", use_container_width=True)
        st.success(f"マッチ成功！検出された位置にラベルを描画しました。")
    else:
        st.warning("マッチングに失敗しました。特徴点が不足している可能性があります。テンプレートや構造図の品質を見直してみてください。")
