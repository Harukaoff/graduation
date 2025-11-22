import streamlit as st
import cv2
import numpy as np

st.title("ORB + BFMatcher によるテンプレートマッチング＆ラベリング")

uploaded_img = st.file_uploader("構造図画像をアップロード", type=["jpg", "jpeg", "png"])
uploaded_template = st.file_uploader("テンプレート画像をアップロード", type=["jpg", "jpeg", "png"])

label_name = st.text_input("ラベルを入力（例：支点A、荷重1）", "マッチ")

if uploaded_img and uploaded_template:
    # 画像読み込み
    img_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    templ_bytes = np.asarray(bytearray(uploaded_template.read()), dtype=np.uint8)
    templ = cv2.imdecode(templ_bytes, cv2.IMREAD_COLOR)

    # グレースケール変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

    # ORB特徴検出器
    orb = cv2.ORB_create(nfeatures=1000)

    kp1, des1 = orb.detectAndCompute(templ_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    # BFMatcher (Hamming距離)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 最良マッチの座標を取得
    if matches:
        best_match = matches[0]
        pt_img = kp2[best_match.trainIdx].pt
        x, y = int(pt_img[0]), int(pt_img[1])

        # マッチ位置にラベル描画
        labeled = img.copy()
        cv2.circle(labeled, (x, y), 15, (0, 255, 0), 3)
        cv2.putText(labeled, label_name, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Streamlit表示
        st.image(cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB), caption="ラベリング結果", use_column_width=True)

        st.success(f"マッチ位置: ({x}, {y})")
    else:
        st.error("マッチが見つかりませんでした。テンプレ画像が小さすぎるか、特徴が少ない可能性があります。")
