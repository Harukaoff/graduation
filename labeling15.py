import sys
import os
import streamlit as st
import numpy as np
import cv2
import math # PCA描画関数で使われているので残します

st.set_page_config(layout="wide") # レイアウトを広めに設定

st.title("構造図の要素認識 (テンプレートマッチング特化)")
st.header("1. 構造要素の画像認識")

# --- テンプレートの読み込み ---
# 'templates'フォルダがスクリプトと同じディレクトリにあると仮定します
# 各ラベルに対して複数のテンプレートファイルパスを指定できるように変更
template_files = {
    "pin": ["templates/pin1.png","templates/pin2.png","templates/pin3.png","templates/pin4.png","templates/pin5.png","templates/pin6.png"], # 例: 複数のピンテンプレート
    "roller": ["templates/roller1.png","templates/roller2.png","templates/roller3.png","templates/roller4.png","templates/roller5.png","templates/roller6.png",],
    "fixed": ["templates/fixed1.png","templates/fixed2.png","templates/fixed3.png","templates/fixed4.png","templates/fixed5.png","templates/fixed6.png"],
    "hinge": ["templates/hinge1.png","templates/hinge2.png","templates/hinge3.png","templates/hinge4.png","templates/hinge5.png","templates/hinge6.png",],
    "weight": ["templates/weight1.png","templates/weight2.png","templates/weight3.png","templates/weight4.png","templates/weight5.png","templates/weight6.png"]
}
# テンプレート画像をロードし、各ラベルに対してリストで保持
templates_loaded = {}
for label, paths in template_files.items():
    loaded_list = []
    for path in paths:
        try:
            templ = cv2.imread(path, 0)
            if templ is not None:
                loaded_list.append(templ)
            else:
                st.warning(f"テンプレート画像が見つからないか、読み込めません: {path}")
        except Exception as e:
            st.error(f"テンプレート画像読み込みエラー {path}: {e}")
    if loaded_list:
        templates_loaded[label] = loaded_list

if not templates_loaded:
    st.error("テンプレート画像が一つも読み込めませんでした。'templates'フォルダ内の画像ファイルを確認してください。")
    st.stop() # テンプレートがないと実行できないため停止

# --- 輪郭検出関数 ---
def detect_contours(gray_img, min_area, max_area):
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area], thr

# --- PCA描画関数 ---
def draw_pca(img_draw, cnt_data):
    if len(cnt_data) < 2:
        return img_draw
    data = cnt_data.reshape(-1, 2).astype(np.float32)
    try:
        mean, eigen = cv2.PCACompute(data, mean=None)
        center = tuple(mean[0].astype(int))
        direction = eigen[0] * 100
        end = (int(center[0] + direction[0]), int(center[1] + direction[1]))
        cv2.line(img_draw, center, end, (255, 0, 0), 2)
    except cv2.error:
        pass 
    return img_draw

# --- テンプレートマッチング関数 (マルチスケール対応) ---
def match_template_region(roi_img):
    best_label, best_score = None, -1
    
    # テンプレートのスケールを試す範囲
    scales = np.linspace(0.5, 1.5, 20) # 0.5倍から1.5倍まで20段階で試す

    for label, template_list in templates_loaded.items():
        for templ in template_list: # 各ラベルの複数のテンプレートをループ
            if templ is None:
                continue

            for scale in scales:
                template_h, template_w = templ.shape[0], templ.shape[1]
                new_w = int(template_w * scale)
                new_h = int(template_h * scale)

                if new_w > roi_img.shape[1] or new_h > roi_img.shape[0] or new_w == 0 or new_h == 0:
                    continue

                resized_templ = cv2.resize(templ, (new_w, new_h), interpolation=cv2.INTER_AREA)

                try:
                    if roi_img.shape[0] < resized_templ.shape[0] or roi_img.shape[1] < resized_templ.shape[1]:
                        continue

                    res = cv2.matchTemplate(roi_img, resized_templ, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    score = max_val

                    if score > best_score:
                        best_label, best_score = label, score
                except cv2.error:
                    pass 
    return best_label, best_score

# --- Streamlit アプリのメイン処理 ---
uploaded_file = st.file_uploader("構造図の画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file:
    buf = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    src_img = cv2.imdecode(buf, 1)
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    output_img = src_img.copy()

    st.sidebar.header("画像認識設定")
    min_contour_area = st.sidebar.number_input("輪郭の最小面積", min_value=10, value=100)
    max_contour_area = st.sidebar.number_input("輪郭の最大面積", min_value=100, value=5000)

    cnts, thr_img = detect_contours(gray_img, min_contour_area, max_contour_area)
    st.write(f"検出された輪郭数: {len(cnts)}")
    st.image(thr_img, caption='二値化画像', use_column_width=True)

    detected_elements_info = []
    
    match_threshold = st.slider("マッチングの信頼度閾値", 0.0, 1.0, 0.5, 0.05)

    count = 0
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue

        roi = gray_img[y:y+h, x:x+w]
        label, score = match_template_region(roi)

        if label and score > match_threshold:
            count += 1
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} {score:.2f} ({x},{y})"
            cv2.putText(output_img, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            draw_pca(output_img, cnt)
            detected_elements_info.append(f"- **{label}**: 信頼度 {score:.2f}, 位置 (x={x}, y={y})")
        else:
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(output_img, "?", (x+2, y+h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB),
              caption=f"検出された構造要素: {count}個", use_column_width=True)

    if detected_elements_info:
        st.write("### 検出された要素の詳細:")
        for info in detected_elements_info:
            st.markdown(info)
    else:
        st.info("画像から有効な構造要素は見つかりませんでした。テンプレートや画像のコントラスト、または閾値を確認してください。")

st.markdown("---")
st.info("""
**このアプリはテンプレートマッチングの精度をテストするためのものです。**

* **テンプレート画像の準備:** `templates` フォルダ内に、検出したい記号の画像を複数枚（例: `pin1.png`, `pin2.png`）用意することで、より高い精度が期待できます。手書きの図面を扱う場合は、様々な手書きのバリエーションをテンプレートとして用意することが非常に重要です。
* **輪郭の最小/最大面積:** サイドバーの「輪郭の最小面積」と「輪郭の最大面積」を調整することで、検出対象となる記号のサイズ範囲を絞り込むことができます。
* **マッチングの信頼度閾値:** スライダーで調整できるこの閾値を上げることで、誤検出を減らすことができますが、その反面、見落としも増える可能性があります。
""")