import cv2
import numpy as np
import streamlit as st

st.title("輪郭認識(PCA) + テンプレートマッチ でラベリング")

# テンプレートの読み込み
template_files = {
    "ピン": "templates/pin2.png",
    "ローラー": "templates/roller2.png",
    "固定": "templates/fixed1.png",
    "ヒンジ": "templates/hinge.png",
    "荷重": "templates/kajyu.png"
}
templates = {label: cv2.imread(path, 0) for label, path in template_files.items()}

def detect_contours(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 5000], thr

def draw_pca(img, cnt):
    data = cnt.reshape(-1, 2).astype(np.float32)
    mean, eigen = cv2.PCACompute(data, mean=None)
    center = tuple(mean[0].astype(int))
    direction = eigen[0] * 100
    end = (int(center[0] + direction[0]), int(center[1] + direction[1]))
    cv2.line(img, center, end, (255, 0, 0), 2)
    return img

def match_template_region(roi):
    best_label, best_score = None, -1
    for label, templ in templates.items():
        try:
            roi_resize = cv2.resize(roi, (templ.shape[1], templ.shape[0]))
            res = cv2.matchTemplate(roi_resize, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            if score > best_score:
                best_label, best_score = label, score
        except:
            pass
    return best_label, best_score

uploaded = st.file_uploader("構造図アップロード", type=["jpg", "png", "jpeg"])

if uploaded:
    buf = np.frombuffer(uploaded.read(), dtype=np.uint8)
    src = cv2.imdecode(buf, 1)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    output = src.copy()

    cnts, thr = detect_contours(gray)
    st.write(f"検出された輪郭数: {len(cnts)}")
    st.image(thr, caption='二値化画像', use_column_width=True)

    count = 0
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y:y+h, x:x+w]
        label, score = match_template_region(roi)

        if label and score > 0.3:
            count += 1
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output, f"{label} {score:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            draw_pca(output, cnt)
        else:
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(output, "?", (x+2, y+h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
             caption=f"検出: {count}個", use_column_width=True)

    if count == 0:
        st.info("有効な構造要素は見つかりませんでした。テンプレートや画像のコントラストを確認してください。")
