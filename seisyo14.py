import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image


# ===============================
# テンプレート読み込み
# ===============================
def load_template(path):
    t = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if t is None:
        st.warning(f"テンプレートが読み込めません: {path}")
        return None
    if t.shape[2] == 3:
        b, g, r = cv2.split(t)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        t = cv2.merge([b, g, r, alpha])
    return t


templates = {
    "pin": load_template("templates/pin.png"),
    "roller": load_template("templates/roller.png"),
    "fixed": load_template("templates/fixed.png"),
    "beam": load_template("templates/beam.png"),
    "load": load_template("templates/load.png"),
    "moment l": load_template("templates/momentL.png"),
    "moment r": load_template("templates/momentR.png"),
    "udl": load_template("templates/UDL.png"),
}


# ===============================
# 合成関数
# ===============================
def overlay_template(base_img, template_img, center, angle, scale=1.0):
    h, w = template_img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    new_w, new_h = max(new_w, 5), max(new_h, 5)

    # テンプレートの回転
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        cv2.resize(template_img, (new_w, new_h)),
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

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


# ===============================
# メインアプリ
# ===============================
def run_app():
    st.title("🏗️ 構造図 清書アプリ（梁の縦方向対応ver）")

    model_path = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
    if not os.path.exists(model_path):
        st.error("モデルファイルが見つかりません。")
        st.stop()

    model = YOLO(model_path)
    conf_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.05)
    uploaded_file = st.file_uploader("構造図をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        st.image(img_pil, caption="アップロード画像", use_container_width=True)

        if st.button("清書を実行"):
            with st.spinner("検出・清書中..."):
                results = model(img, conf=conf_th, imgsz=640)[0]
                canvas = np.ones_like(img) * 255
                beams = []

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
                        angle_deg = -math.degrees(angle)
                        angle_deg = ((angle_deg + 180) % 360) - 180
                        angle_deg = round(angle_deg / 30) * 30  # 30°刻み

                        # スケール補正
                        if "beam" in name:
                            # 梁の長さ方向でスケール計算（縦梁対策）
                            long_side = max(w, h)
                            scale = long_side / templates[name].shape[1]
                            scale = max(scale, 0.3)
                            beams.append((x, y, w, h, angle_deg))
                        else:
                            scale = max(w / templates[name].shape[1], 0.3)

                        st.text(f"{name}: angle={angle_deg:.1f}° conf={conf:.2f}")
                        canvas = overlay_template(canvas, templates[name], (x, y), angle_deg, scale)

                

                img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="清書結果", use_container_width=True)

    else:
        st.info("構造図（png/jpg）をアップロードしてください。")


if __name__ == "__main__":
    run_app()
