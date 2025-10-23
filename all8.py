import os
import math
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ==========================================================
# 剛性マトリクス関連関数（はるのコード）
# ==========================================================
def make_T3(angle):
    mu = math.sin(math.radians(angle))
    lamb = math.cos(math.radians(angle))
    T3 = np.array([
        [lamb, mu, 0, 0, 0, 0],
        [-mu, lamb, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, lamb, mu, 0],
        [0, 0, 0, -mu, lamb, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    return T3

def esm(E, A, I, L, angle):
    matrix_L = np.array([
        [(E*A)/L, 0, 0, -(E*A)/L, 0, 0],
        [0, (12*E*I)/L**3, (6*E*I)/L**2, 0, -(12*E*I)/L**3, (6*E*I)/L**2],
        [0, (6*E*I)/L**2, (4*E*I)/L, 0, -(6*E*I)/L**2, (2*E*I)/L],
        [-(E*A)/L, 0, 0, (E*A)/L, 0, 0],
        [0, -(12*E*I)/L**3, -(6*E*I)/L**2, 0, (12*E*I)/L**3, -(6*E*I)/L**2],
        [0, (6*E*I)/L**2, (2*E*I)/L, 0, -(6*E*I)/L**2, (4*E*I)/L]
    ])
    matrix_T3 = make_T3(angle)
    matrix_G = np.dot(matrix_T3.T, np.dot(matrix_L, matrix_T3))
    return matrix_G

# ==========================================================
# YOLO + テンプレート清書部
# ==========================================================
def load_template(path):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if temp is None:
        st.warning(f"テンプレートが読み込めません: {path}")
        return None
    if temp.shape[2] == 3:
        b, g, r = cv2.split(temp)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        temp = cv2.merge([b, g, r, alpha])
    return temp

# テンプレート登録
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

def overlay_template(base_img, template_img, center, angle, scale=1.0):
    """テンプレートを回転・スケーリングして貼り付け"""
    h, w = template_img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return base_img
    template_resized = cv2.resize(template_img, (new_w, new_h))
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

# ==========================================================
# Streamlit アプリ本体
# ==========================================================
def run_app():
    st.title("🏗️ 構造図 自動認識＋清書＋構造解析アプリ")
    st.write("YOLOv8で手書き図を解析し、清書 → 剛性マトリクス法で解析まで行います。")

    model_path = "runs/obb/train28/weights/best.pt"
    if not os.path.exists(model_path):
        st.error("モデルファイルが見つかりません。")
        st.stop()
    model = YOLO(model_path)

    conf_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.05)

    uploaded_file = st.file_uploader("構造図画像をアップロード", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        st.image(img_pil, caption="アップロード画像", use_container_width=True)

        if st.button("検出＆清書"):
            with st.spinner("YOLOで要素を検出中..."):
                results = model(img, conf=conf_th, imgsz=640)[0]

                # 白紙キャンバス作成
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
                        template = templates[name]
                        scale = max(w / template.shape[1], 0.1)
                        canvas = overlay_template(canvas, template, (x, y), angle_deg, scale)
                        elem_info.append({
                            "class": name, "x": x, "y": y,
                            "w": w, "h": h, "angle": angle_deg
                        })
                        cv2.putText(canvas, name, (int(x), int(y)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="清書結果", use_container_width=True)

                # --- 構造解析の簡易実行 ---
                if st.button("剛性マトリクス法で解析実行"):
                    st.write("（ここに後で剛性マトリクス法＋応力図出力を統合）")
                    st.json(elem_info)  # とりあえず要素情報を出力確認

                # 保存とDL
                os.makedirs("output", exist_ok=True)
                result_pil = Image.fromarray(img_rgb)
                result_pil.save("output/cleaned.png")
                with open("output/cleaned.png", "rb") as f:
                    st.download_button("清書画像をダウンロード", f, file_name="cleaned.png")

    else:
        st.info("構造図をアップロードしてください。")

if __name__ == "__main__":
    run_app()
