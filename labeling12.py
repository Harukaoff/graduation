import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="テンプレートラベリング", layout="wide")
st.title("🔍 ORB + BFMatcher による構造図テンプレートラベリング（特徴点拡張版）")

# --- 画像アップロード ---
st.sidebar.header("画像アップロード")
uploaded_img = st.sidebar.file_uploader("構造図画像", type=["jpg", "jpeg", "png"])
uploaded_template = st.sidebar.file_uploader("テンプレート画像", type=["jpg", "jpeg", "png"])

label_name = st.sidebar.text_input("🔖 ラベル名", "マッチ")

if uploaded_img and uploaded_template:
    img_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    templ_bytes = np.frombuffer(uploaded_template.read(), np.uint8)
    templ = cv2.imdecode(templ_bytes, cv2.IMREAD_COLOR)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

    # --- テンプレート前処理 ---
    templ_gray = cv2.GaussianBlur(templ_gray, (3, 3), 0)
    templ_gray = cv2.Canny(templ_gray, 50, 150)

    # --- ORB特徴点抽出（特徴点多めに） ---
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(templ_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    st.subheader("🔑 特徴点の検出")
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.drawKeypoints(templ, kp1, None, color=(0,255,0)), caption="テンプレートの特徴点", use_column_width=True)
    with col2:
        st.image(cv2.drawKeypoints(img, kp2, None, color=(255,0,0)), caption="構造図の特徴点", use_column_width=True)

    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # 一定距離以下だけ採用
        good_matches = [m for m in matches if m.distance < 60]

        if good_matches:
            best = good_matches[0]
            pt_img = kp2[best.trainIdx].pt
            x, y = int(pt_img[0]), int(pt_img[1])

            labeled_img = img.copy()
            cv2.circle(labeled_img, (x, y), 15, (0, 255, 0), 3)
            cv2.putText(labeled_img, label_name, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            st.subheader("✅ ラベリング結果")
            st.image(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB), caption=f"位置: ({x}, {y})", use_column_width=True)

            match_vis = cv2.drawMatches(templ, kp1, img, kp2, good_matches[:10], None, flags=2)
            st.subheader("📎 マッチ結果 (上位10件)")
            st.image(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            st.warning("特徴点は検出されたけど、有効なマッチが見つかりませんでした。テンプレ画像の調整を試してください。")
    else:
        st.error("😢 特徴点が検出できませんでした。テンプレート画像をもっとくっきりしたものにしてみてください。")
else:
    st.info("左のサイドバーから画像とテンプレートをアップロードしてください。")
