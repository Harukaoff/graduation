import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image


# ======================
# オーバーレイ関数
# ======================
def overlay_template(base_img, template_img, center, angle, scale=1.0):
    h, w = template_img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    new_w, new_h = max(new_w, 5), max(new_h, 5)

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


# ======================
# 点の順序をクラス別に補正
# ======================
def reorder_points(pts, cls_name):
    # 上→右→下→左（基本的な順）
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    if "beam" in cls_name:
        # 1-2, 3-4 が長辺
        d01 = np.linalg.norm(pts[0] - pts[1])
        d12 = np.linalg.norm(pts[1] - pts[2])
        if d01 < d12:
            pts = np.roll(pts, -1, axis=0)

    elif "load" in cls_name:
        # 1-2, 3-4 が長辺、2-3 に矢じり
        dists = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        long_idx = np.argmax(dists)
        pts = np.roll(pts, -long_idx, axis=0)

    elif "pin" in cls_name or "roller" in cls_name:
        # 1-2 の間に支点先端がある想定（尖った方向が上や下）
        top_idx = np.argmin(pts[:, 1])  # 最も上の点を1に
        pts = np.roll(pts, -top_idx, axis=0)

    return pts


# ======================
# Streamlitアプリ
# ======================
def main():
    st.title("🏗️ YOLOv8-OBB 構造図 検出＋清書（点順補正付き）")

    model_path = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
    if not os.path.exists(model_path):
        st.error("モデルファイルが見つかりません。")
        st.stop()

    model = YOLO(model_path)
    conf_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.05)
    uploaded_file = st.file_uploader("構造図をアップロード", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        img = np.array(img_pil)
        st.image(img, caption="📥 アップロード画像", use_container_width=True)

        if st.button("検出と清書を実行"):
            with st.spinner("検出中..."):
                results = model(img)
                result = results[0]

            vis_img = img.copy()
            clean_img = np.ones_like(img) * 255

            if result.obb is not None:
                for i in range(len(result.obb.cls)):
                    cls_id = int(result.obb.cls[i])
                    conf = float(result.obb.conf[i])
                    if conf < conf_th:
                        continue

                    name = model.names[cls_id].lower()
                    pts = result.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                    pts = reorder_points(pts, name)

                    # 表示
                    cv2.polylines(vis_img, [pts.astype(np.int32)], True, (0, 255, 0), 2)
                    for j, (x, y) in enumerate(pts):
                        cv2.circle(vis_img, (int(x), int(y)), 4, (0, 0, 255), -1)
                        cv2.putText(vis_img, str(j + 1), (int(x) + 5, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # 角度計算
                    v1 = pts[1] - pts[0]
                    angle = math.degrees(math.atan2(v1[1], v1[0]))

                    # テンプレート
                    path = f"templates/{name}.png"
                    if not os.path.exists(path):
                        st.warning(f"未対応: {name}")
                        continue
                    tpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if tpl is None:
                        continue

                    # スケールと貼り付け
                    w = np.linalg.norm(pts[0] - pts[1])
                    h = np.linalg.norm(pts[1] - pts[2])
                    scale = max(w / tpl.shape[1], h / tpl.shape[0], 0.5)
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                    clean_img = overlay_template(clean_img, tpl, (cx, cy), angle, scale)

            st.image([img, vis_img, clean_img],
                     caption=["入力画像", "点順補正付きBBox", "清書結果"],
                     use_container_width=True)


if __name__ == "__main__":
    main()
