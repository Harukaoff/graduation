import streamlit as st
import cv2
import numpy as np
import math
from ultralytics import YOLO

# ======================
# テンプレートの読み込み
# ======================
templates = {
    "pin": cv2.imread("templates/pin.jpg", cv2.IMREAD_UNCHANGED),
    "roller": cv2.imread("templates/roller.jpg", cv2.IMREAD_UNCHANGED),
    "fixed": cv2.imread("templates/fixed.jpg", cv2.IMREAD_UNCHANGED),
    "beam": cv2.imread("templates/beam.jpg", cv2.IMREAD_UNCHANGED),
    "load": cv2.imread("templates/load.jpg", cv2.IMREAD_UNCHANGED),
    "moment_l": cv2.imread("templates/momentL.jpg", cv2.IMREAD_UNCHANGED),
    "moment_r": cv2.imread("templates/momentR.jpg", cv2.IMREAD_UNCHANGED),
    "udl": cv2.imread("templates/UDL.jpg", cv2.IMREAD_UNCHANGED),
}

# ======================
# ポリゴンから角度を算出
# ======================
def get_angle_from_polygon(pts):
    edges = []
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        length = math.hypot(x2 - x1, y2 - y1)
        edges.append(((x1, y1), (x2, y2), length))
    edges = sorted(edges, key=lambda e: e[2], reverse=True)
    (x1, y1), (x2, y2), _ = edges[0]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

# ======================
# 角度を15°刻みにスナップ
# ======================
def snap_to_15(angle):
    return round(angle / 15) * 15

# ======================
# テンプレートを回転して重ねる（縦横比固定）
# ======================
def overlay_template(base_img, template, pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return base_img

    th, tw = template.shape[:2]
    scale = min(w / tw, h / th)  # アスペクト比を保持
    new_w, new_h = int(tw * scale), int(th * scale)
    resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 角度計算してスナップ
    angle = snap_to_15(get_angle_from_polygon(pts))

    # 回転
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        resized, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0)
    )

    # 貼り付け位置（中心合わせ）
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1_new, y1_new = cx - new_w // 2, cy - new_h // 2
    x2_new, y2_new = x1_new + new_w, y1_new + new_h

    # 範囲クリップ
    x1_new, y1_new = max(0, x1_new), max(0, y1_new)
    x2_new, y2_new = min(base_img.shape[1], x2_new), min(base_img.shape[0], y2_new)

    roi = base_img[y1_new:y2_new, x1_new:x2_new]
    if roi.size == 0:
        return base_img

    overlay = rotated[:, :, :3]
    mask = rotated[:, :, 3] if rotated.shape[2] == 4 else cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(overlay, overlay, mask=mask)
    combined = cv2.add(bg, fg)

    base_img[y1_new:y2_new, x1_new:x2_new] = combined
    return base_img

# ======================
# Streamlit アプリ
# ======================
st.title("手書き構造図の清書アプリ（信頼度フィルタ付き・15°スナップ）")

uploaded_file = st.file_uploader("構造図画像をアップロードしてください", type=["jpg", "jpeg", "png"])
confidence_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.05)

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    model = YOLO("runs/obb/train7/weights/best.pt")

    results = model.predict(img, conf=confidence_th)

    for r in results:
        if hasattr(r, "obb") and r.obb is not None:
            for i, box in enumerate(r.obb.xyxyxyxy):
                cls = int(r.obb.cls[i])
                conf = float(r.obb.conf[i])
                if conf < confidence_th:
                    continue  # 一定の信頼度未満はスキップ

                name = model.names[cls]
                pts = [(int(x), int(y)) for x, y in box.reshape(-1, 2)]

                if name in templates and templates[name] is not None:
                    img = overlay_template(img, templates[name], pts)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="清書結果", use_container_width=True)
    save_path = "output/seisyo_result.jpg"
    cv2.imwrite(save_path, img)
    st.success(f"清書画像を保存しました: {save_path}")
