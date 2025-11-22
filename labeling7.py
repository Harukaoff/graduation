import streamlit as st
import cv2
import numpy as np

st.title("AKAZE + BFMatcher によるテンプレートマッチング＆ラベリング")

uploaded_img = st.file_uploader("構造図画像をアップロード", type=["jpg", "jpeg", "png"])
uploaded_template = st.file_uploader("テンプレート画像をアップロード", type=["jpg", "jpeg", "png"])

label_name = st.text_input("ラベルを入力（例：支点A、荷重1）", "マッチ")

if uploaded_img and uploaded_template:
    # OpenCV形式に変換
    img_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    templ_bytes = np.asarray(bytearray(uploaded_template.read()), dtype=np.uint8)
    templ = cv2.imdecode(templ_bytes, cv2.IMREAD_COLOR)

    # グレースケール変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

    # AKAZE特徴量抽出
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(templ_gray, None)
    kp2, des2 = akaze.detectAndCompute(img_gray, None)

    # BFMatcher (Hamming)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # ベストマッチ1個の座標
        best = matches[0]
        pt_img = kp2[best.trainIdx].pt
        x, y = int(pt_img[0]), int(pt_img[1])

        labeled_img = img.copy()
        cv2.circle(labeled_img, (x, y), 15, (0, 255, 0), 3)
        cv2.putText(labeled_img, label_name, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        st.image(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB),
                 caption="ラベリング結果", use_column_width=True)
        st.success(f"マッチ成功！座標: ({x}, {y})")
    else:
        st.error("特徴点が検出できませんでした。テンプレートをもっとくっきりしたものにしてみて！")
