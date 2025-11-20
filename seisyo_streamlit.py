import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import math
import os

# ==========================
# 初期設定
# ==========================
st.set_page_config(page_title="構造図 自動清書", layout="wide")

MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
TEMPLATE_DIR = r"C:\Users\morim\Downloads\graduation\templates"

# ==========================
# テンプレート読み込み関数
# ==========================
def load_template(path):
    t = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if t is None:
        st.warning(f"テンプレート読み込み失敗: {path}")
        return None
    if t.shape[2] == 3:
        b, g, r = cv2.split(t)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        t = cv2.merge([b, g, r, alpha])
    return t

templates = {
    "pin": load_template(os.path.join(TEMPLATE_DIR, "pin.png")),
    "roller": load_template(os.path.join(TEMPLATE_DIR, "roller.png")),
    "fixed": load_template(os.path.join(TEMPLATE_DIR, "fixed.png")),
    "beam": load_template(os.path.join(TEMPLATE_DIR, "beam.png")),
    "load": load_template(os.path.join(TEMPLATE_DIR, "load.png")),
    "moment l": load_template(os.path.join(TEMPLATE_DIR, "momentL.png")),
    "moment r": load_template(os.path.join(TEMPLATE_DIR, "momentR.png")),
    "udl": load_template(os.path.join(TEMPLATE_DIR, "UDL.png")),
}

# ==========================
# ユーティリティ
# ==========================
def overlay_template(base_img, template_img, center, angle, scale=1.0):
    """透過付きテンプレートを回転して合成"""
    h, w = template_img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return base_img
    template_resized = cv2.resize(template_img, (new_w, new_h))

    # 回転行列
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
    rotated = cv2.warpAffine(template_resized, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    x, y = int(center[0]), int(center[1])
    x1, y1 = max(x - new_w // 2, 0), max(y - new_h // 2, 0)
    x2, y2 = min(x1 + new_w, base_img.shape[1]), min(y1 + new_h, base_img.shape[0])

    roi = base_img[y1:y2, x1:x2]
    rot_crop = rotated[0:(y2 - y1), 0:(x2 - x1)]

    alpha = rot_crop[:, :, 3] / 255.0
    for c in range(3):
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * rot_crop[:, :, c]
    base_img[y1:y2, x1:x2] = roi
    return base_img


# ==========================
# Streamlit UI
# ==========================
st.title("🏗️ 構造図 清書＋テンプレート配置ツール")

model = YOLO(MODEL_PATH)

uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])
conf_th = st.slider("信頼度しきい値 (Confidence Threshold)", 0.0, 1.0, 0.4, 0.05)
scale_factor = st.slider("テンプレート倍率 (scale)", 0.3, 2.0, 1.0, 0.1)

if uploaded_file:
    # 入力画像
    img_pil = Image.open(uploaded_file).convert("RGB")
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    st.image(img_pil, caption="入力画像", use_container_width=True)

    if st.button("清書を実行"):
        with st.spinner("YOLOv8-OBB 推論中..."):
            results = model(img, conf=conf_th, imgsz=640)[0]

        canvas = np.ones_like(img) * 255
        elem_info = []

        if hasattr(results, "obb") and results.obb is not None:
            for box in results.obb:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                if conf < conf_th:
                    continue
                name = results.names[cls_id]
                if name not in templates or templates[name] is None:
                    continue

                x, y, w, h, angle = box.xywhr.cpu().numpy()[0]
                angle_deg = -float(angle * 180 / np.pi)

                # ===== 梁の縦横補正 =====
                if "beam" in name:
                    if abs(angle_deg) > 45 and abs(angle_deg) < 135:
                        angle_deg += 90  # 縦梁補正

                # ===== モーメント向き補正 =====
                if "moment" in name:
                    angle_deg = round(angle_deg / 15) * 15  # スナップ

                # ===== UDLは水平に近づける =====
                if "udl" in name:
                    if abs(angle_deg) > 45:
                        angle_deg = 90

                # ===== 貼り付け =====
                template = templates[name]
                scale = scale_factor
                canvas = overlay_template(canvas, template, (x, y), angle_deg, scale)
                elem_info.append({"class": name, "x": x, "y": y, "angle": angle_deg})

        st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption="清書結果", use_container_width=True)

        # 検出情報のテーブル出力
        if elem_info:
            st.subheader("検出された要素一覧")
            st.dataframe(elem_info)
        else:
            st.warning("要素が検出されませんでした。閾値を下げて再試行してください。")

else:
    st.info("PNGまたはJPG画像をアップロードしてください。")
