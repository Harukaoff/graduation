import streamlit as st
import cv2
import numpy as np
import os

st.title("AKAZE特徴点マッチング + スコアでラベリング")

uploaded_image = st.file_uploader("構造図全体画像をアップロード", type=["jpg", "jpeg", "png"])

# テンプレートファイルと対応するラベル名
template_files = {
    "C:/Users/morim/Downloads/graduation/templates/pin2.png": "pin",
    "C:/Users/morim/Downloads/graduation/templates/roller2.png": "roller",
    "C:/Users/morim/Downloads/graduation/templates/kajyu.png": "weight",
    "C:/Users/morim/Downloads/graduation/templates/fixed1.png": "fixed"
}



# 上位何件のマッチを取るか
TOP_K = 2

if uploaded_image:
    # 全体画像読み込み
    img_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # マッチ結果を格納するリスト
    all_matches = []

    # 各テンプレートに対して処理
    for template_path, label_prefix in template_files.items():
        if not os.path.exists(template_path):
            st.warning(f"テンプレートが見つかりません: {template_path}")
            continue

        templ = cv2.imread(template_path)
        templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(templ_gray, None)
        kp2, des2 = akaze.detectAndCompute(img_gray, None)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # 距離が短い（スコアが高い）順にソート
        matches = sorted(matches, key=lambda x: x.distance)[:TOP_K]

        for idx, m in enumerate(matches):
            pt = kp2[m.trainIdx].pt
            all_matches.append({
                "x": int(pt[0]),
                "y": int(pt[1]),
                "score": m.distance,
                "label": f"{label_prefix}{idx+1}"
            })

    # 重複を削除（同じような位置のマッチに対して、スコアが高い方を残す）
    final_matches = []
    threshold_dist = 30  # ピクセル距離が近すぎると同一とみなす

    for match in sorted(all_matches, key=lambda x: x["score"]):
        is_duplicate = False
        for fm in final_matches:
            dist = np.hypot(fm["x"] - match["x"], fm["y"] - match["y"])
            if dist < threshold_dist:
                is_duplicate = True
                break
        if not is_duplicate:
            final_matches.append(match)

    # 結果描画
    result_img = img.copy()
    for match in final_matches:
        cv2.circle(result_img, (match["x"], match["y"]), 20, (0, 255, 0), 3)
        cv2.putText(result_img, match["label"], (match["x"] + 10, match["y"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
             caption="ラベリング結果", use_column_width=True)

    if not final_matches:
        st.warning("一致する要素が見つかりませんでした。テンプレートや画像の画質・向きを確認してね。")
