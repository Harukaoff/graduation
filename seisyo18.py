import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image


# ======================
# テンプレート読み込み関数
# ======================
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


# ======================
# テンプレート登録
# ======================
templates = {
    "pin": load_template("templates/pin.png"),
    "roller": load_template("templates/roller.png"),
    "hinge": load_template("templates/hinge.png"),
    "fixed": load_template("templates/fixed.png"),
    "beam": load_template("templates/beam.png"),
    "load": load_template("templates/load.png"),
    "momentl": load_template("templates/momentL.png"),
    "momentr": load_template("templates/momentR.png"),
    "udl": load_template("templates/UDL.png"),
}


# ======================
# オーバーレイ関数
# ======================
def overlay_template(base_img, template_img, center, angle, scale=1.0):
    h, w = template_img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    new_w, new_h = max(new_w, 5), max(new_h, 5)

    resized = cv2.resize(template_img, (new_w, new_h))

    M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    rot_w = int(new_h * sin + new_w * cos)
    rot_h = int(new_h * cos + new_w * sin)

    M[0, 2] += (rot_w / 2) - new_w / 2
    M[1, 2] += (rot_h / 2) - new_h / 2

    rotated = cv2.warpAffine(
        resized, M, (rot_w, rot_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    x, y = int(center[0]), int(center[1])
    x1, y1 = x - rot_w // 2, y - rot_h // 2
    x2, y2 = x1 + rot_w, y1 + rot_h

    if x2 < 0 or y2 < 0 or x1 >= base_img.shape[1] or y1 >= base_img.shape[0]:
        return base_img

    x1_clip, y1_clip = max(0, x1), max(0, y1)
    x2_clip, y2_clip = min(base_img.shape[1], x2), min(base_img.shape[0], y2)
    rot_crop = rotated[
        (y1_clip - y1):(y2_clip - y1),
        (x1_clip - x1):(x2_clip - x1)
    ]

    alpha = rot_crop[:, :, 3] / 255.0
    roi = base_img[y1_clip:y2_clip, x1_clip:x2_clip]

    for c in range(3):
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * rot_crop[:, :, c]
    base_img[y1_clip:y2_clip, x1_clip:x2_clip] = roi
    return base_img


# ======================
# Streamlit アプリ本体
# ======================
def run_app():
    st.title("🏗️ 構造図 清書アプリ（検出＋清書＋可視化付き）")

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
        st.image(img_pil, caption="📥 アップロード画像", use_container_width=True)

        if st.button("検出と清書を実行"):
            with st.spinner("検出中..."):
                results = model(img, conf=conf_th, imgsz=640)[0]
                vis_img = img.copy()

                # ====== 検出結果の可視化 ======
                if hasattr(results, "obb") and results.obb is not None:
                    for box in results.obb:
                        xyxyxyxy = box.xyxyxyxy.cpu().numpy()[0].reshape((-1, 1, 2)).astype(np.int32)
                        cv2.polylines(vis_img, [xyxyxyxy], True, (0, 0, 255), 2)

                        cls_id = int(box.cls.cpu().numpy()[0])
                        conf = float(box.conf.cpu().numpy()[0])
                        x, y, w, h, angle = box.xywhr.cpu().numpy()[0]
                        name = results.names[cls_id]
                        cv2.putText(
                            vis_img,
                            f"{name} {conf:.2f}",
                            (int(x - w / 2), int(y - h / 2) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), caption="🔍 検出結果（バウンディングボックス）", use_container_width=True)

            # ====== 清書処理 ======
            with st.spinner("清書中..."):
                canvas = np.ones_like(img) * 255

                if hasattr(results, "obb") and results.obb is not None:
                    for box in results.obb:
                        cls_id = int(box.cls.cpu().numpy()[0])
                        conf = float(box.conf.cpu().numpy()[0])
                        if conf < conf_th:
                            continue

                        name = results.names[cls_id].lower().replace(" ", "")
                        if name not in templates or templates[name] is None:
                            st.text(f"⚠ 未対応クラス: {name}")
                            continue

                        x, y, w, h, angle = box.xywhr.cpu().numpy()[0]
                        angle_deg = -math.degrees(angle)
                        angle_deg = ((angle_deg + 180) % 360) - 180  # 正規化

                        tpl = templates[name]

                        # ===============================
                        # 🎯 梁と固定支点の角度補正
                        # ===============================
                        if "beam" in name:
                            # 縦長なら90°補正
                            if w < h:
                                angle_deg += 90
                            angle_deg = round(angle_deg / 15) * 15
                            scale_w = w / tpl.shape[1]
                            scale_h = h / tpl.shape[0]
                            scale = max(min(scale_w, scale_h), 0.6)

                        elif "fixed" in name:
                            # 横長なら90°補正（←ここを逆にした）
                            if w > h:
                                angle_deg += 90
                            angle_deg = round(angle_deg / 15) * 15
                            scale_w = w / tpl.shape[1]
                            scale_h = h / tpl.shape[0]
                            scale = max(min(scale_w, scale_h), 0.6)

                        else:
                            scale = max(w / tpl.shape[1], 0.4)

                        st.text(f"{name}: angle={angle_deg:.1f}° conf={conf:.2f}")
                        canvas = overlay_template(canvas, tpl, (x, y), angle_deg, scale)

                img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="🧱 清書結果", use_container_width=True)

    else:
        st.info("構造図（png/jpg）をアップロードしてください。")


if __name__ == "__main__":
    run_app()
