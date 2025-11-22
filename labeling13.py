import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="構造図検出と再構成", layout="wide")
st.title("🏗️ 構造図：直線＋支点＋荷重の統合検出 & 再描画")

# --- アップロード ---
st.sidebar.header("画像アップロード")
uploaded_img = st.sidebar.file_uploader("構造図画像 (jpg/png)", type=["jpg", "jpeg", "png"])
templ_pin = st.sidebar.file_uploader("テンプレート: ピン支点", type=["jpg", "jpeg", "png"])
templ_roller = st.sidebar.file_uploader("テンプレート: ローラー支点", type=["jpg", "jpeg", "png"])
templ_load = st.sidebar.file_uploader("テンプレート: 荷重矢印", type=["jpg", "jpeg", "png"])

if uploaded_img:
    # 読み込み
    img_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 梁（直線）検出 ---
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    result_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # --- テンプレートマッチング関数 ---
    def match_template(template_bytes, label, color):
        if template_bytes is None:
            return
        templ_np = np.frombuffer(template_bytes.read(), np.uint8)
        templ = cv2.imdecode(templ_np, cv2.IMREAD_COLOR)
        templ_gray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(img_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        w, h = templ.shape[1], templ.shape[0]
        for pt in zip(*loc[::-1]):
            cv2.rectangle(result_img, pt, (pt[0] + w, pt[1] + h), color, 2)
            cv2.putText(result_img, label, (pt[0], pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # --- 支点・荷重のテンプレートマッチング ---
    match_template(templ_pin, "ピン支点", (0, 255, 0))
    match_template(templ_roller, "ローラー支点", (255, 0, 255))
    match_template(templ_load, "荷重", (0, 0, 255))

    # --- 表示 ---
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="検出＆再描画結果", use_container_width=True)
else:
    st.info("左のサイドバーから構造図画像をアップロードしてください。")
