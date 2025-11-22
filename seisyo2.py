import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ======================
# テンプレート読み込み
# ======================
TEMPLATE_DIR = Path("templates")
template_files = {
    "pin": "pin.jpg",
    "roller": "roller.jpg",
    "fixed": "fixed.jpg",
    "beam": "beam.jpg",
    "load": "load.jpg",
    "moment_l": "momentL.jpg",
    "moment_r": "momentR.jpg",
    "udl": "UDL.jpg",
}

templates = {}
for key, filename in template_files.items():
    path = TEMPLATE_DIR / filename
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        st.warning(f"⚠️ テンプレート {filename} が読み込めませんでした。")
    templates[key] = img


# ======================
# 補助関数
# ======================
def clip_box(box, img_shape):
    """ボックスを画像サイズに収める"""
    x1, y1, x2, y2 = map(int, box)
    h, w = img_shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return x1, y1, x2, y2


def overlay_template(base_img, template, box, alpha=1.0):
    """テンプレートをボックス内に重ねる"""
    x1, y1, x2, y2 = clip_box(box, base_img.shape)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return base_img

    resized = cv2.resize(template, (w, h))
    roi = base_img[y1:y2, x1:x2]

    if resized.shape[2] == 4:  # RGBA
        overlay = resized[:, :, :3]
        mask = resized[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - mask) * roi[:, :, c] + mask * overlay[:, :, c]
    else:
        roi[:] = cv2.addWeighted(roi, 1 - alpha, resized, alpha, 0)

    base_img[y1:y2, x1:x2] = roi
    return base_img


# ======================
# Streamlit アプリ
# ======================
st.title("✏️ 手書き構造図の清書アプリ（OBB対応版）")

uploaded_file = st.file_uploader("構造図画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 入力画像を読み込み
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

    # YOLOモデル読み込み（OBB用）
    model = YOLO("runs/obb/train7/weights/best.pt")

    # 予測
    results = model(img, verbose=False)[0]

    # OBB or BBox を選択
    detections = results.boxes if results.boxes is not None else results.obb

    if detections is not None and len(detections) > 0:
        for box in detections:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name in templates and templates[name] is not None:
                img = overlay_template(img, templates[name], box.xyxy[0])
    else:
        st.warning("⚠️ 検出結果がありませんでした。")

    # 検出枠オプション
    if st.checkbox("検出枠を表示する"):
        if detections is not None and len(detections) > 0:
            for box in detections:
                x1, y1, x2, y2 = clip_box(box.xyxy[0], img.shape)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, model.names[int(box.cls[0])], (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            st.info("検出枠を表示する対象がありません。")

    # 出力表示
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="清書結果", use_container_width=True)

    # 結果データをJSON表示（デバッグ用）
    if st.checkbox("推論結果を表示する"):
        st.json(results.tojson())

    # 保存・ダウンロード
    save_bytes = cv2.imencode(".png", img)[1].tobytes()
    st.download_button(
        label="📥 清書結果をダウンロード",
        data=save_bytes,
        file_name="seisyo_result.png",
        mime="image/png"
    )
