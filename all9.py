import os
import math
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================================
# 剛性マトリクス法 関数群
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
    T3 = make_T3(angle)
    return T3.T @ matrix_L @ T3

def d_r(K, F, fixed_dofs):
    all_dofs = np.arange(len(F))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    d = np.zeros_like(F)
    d[free_dofs] = np.linalg.solve(K_ff, F_f)
    R = K @ d - F
    return d, R

def member_stress(E, I, A, L, angle, d_local):
    T = make_T3(angle)
    d_L = T @ d_local
    M = (E * I / L**2) * np.array([-d_L[2] - d_L[5]])
    return M

# ==========================================================
# YOLO + テンプレート清書部
# ==========================================================
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

def overlay_template(base_img, template_img, center, angle, scale=1.0):
    h, w = template_img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return base_img
    template_resized = cv2.resize(template_img, (new_w, new_h))
    M = cv2.getRotationMatrix2D((new_w//2, new_h//2), angle, 1.0)
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
# 構造解析アプリ本体
# ==========================================================
def run_app():
    st.title("🏗️ 構造図 自動認識＋解析アプリ（完全版）")

    model_path = "runs/obb/train28/weights/best.pt"
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

        if st.button("検出＆解析実行"):
            with st.spinner("検出中..."):
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
                        template = templates[name]
                        scale = max(w / template.shape[1], 0.1)
                        canvas = overlay_template(canvas, template, (x, y), angle_deg, scale)
                        elem_info.append({
                            "class": name, "x": x, "y": y,
                            "angle": angle_deg
                        })

                # 清書結果表示
                img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="清書結果", use_container_width=True)

                # ===========================================
                # 節点・要素自動生成（超簡略モデル）
                # ===========================================
                pins = [(e["x"], e["y"]) for e in elem_info if e["class"] in ["pin", "roller", "fixed"]]
                beams = [(e["x"], e["y"], e["angle"]) for e in elem_info if e["class"] == "beam"]

                if len(pins) < 2 or len(beams) == 0:
                    st.warning("支点または梁の認識が不足しています。")
                    return

                # 節点定義
                n_d = np.array(pins)
                e_l = []
                for i in range(len(pins)-1):
                    e_l.append([i, i+1])

                # 剛性マトリクス作成
                E, A, I = 2e7, 0.03, 0.0002
                num_nodes = len(n_d)
                K = np.zeros((num_nodes*3, num_nodes*3))
                F = np.zeros(num_nodes*3)
                F[-2] = -1000  # 荷重（仮定）

                for i, (n1, n2) in enumerate(e_l):
                    x1, y1 = n_d[n1]
                    x2, y2 = n_d[n2]
                    L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    k = esm(E, A, I, L, angle)
                    dofs = np.array([
                        n1*3, n1*3+1, n1*3+2,
                        n2*3, n2*3+1, n2*3+2
                    ])
                    for a in range(6):
                        for b in range(6):
                            K[dofs[a], dofs[b]] += k[a, b]

                fixed_dofs = [0, 1, 2]  # 最初の支点固定
                d, R = d_r(K, F, fixed_dofs)

                st.subheader("✅ 節点変位 (mm)")
                st.write(np.round(d, 3))

                st.subheader("✅ 支点反力 (N)")
                st.write(np.round(R, 3))

                # モーメント図（簡易表示）
                M_values = []
                for i, (n1, n2) in enumerate(e_l):
                    d_local = np.zeros(6)
                    d_local[0:3] = d[n1*3:n1*3+3]
                    d_local[3:6] = d[n2*3:n2*3+3]
                    x1, y1 = n_d[n1]
                    x2, y2 = n_d[n2]
                    L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    M = member_stress(E, I, A, L, angle, d_local)
                    M_values.append(float(M))

                st.subheader("📈 モーメント図（相対値）")
                fig, ax = plt.subplots()
                ax.plot(range(len(M_values)), M_values, marker="o")
                ax.set_xlabel("部材番号")
                ax.set_ylabel("モーメント")
                st.pyplot(fig)

    else:
        st.info("構造図をアップロードしてください。")

if __name__ == "__main__":
    run_app()
