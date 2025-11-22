import streamlit as st
import cv2
import numpy as np
import math
from ultralytics import YOLO

# ======================
# テンプレート読み込み
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
    # pts: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    edges = []
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        length = math.hypot(x2 - x1, y2 - y1)
        edges.append(((x1, y1), (x2, y2), length))

    # 一番長い辺を採用
    edges = sorted(edges, key=lambda e: e[2], reverse=True)
    (x1, y1), (x2, y2), _ = edges[0]

    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

# ======================
# 角度を15°にスナップ
# ======================
def snap_to_15(angle):
    return round(angle / 15) * 15

# ======================
# OBB座標を使ってテンプレートを重ねる
# ======================
def overlay_template(base_img, template, pts):
    # ポリゴンの外接矩形を取る
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, y_min, x_max, y_max = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    w, h = x_max - x_min, y_max - y_min
    if w <= 0 or h <= 0:
        return base_img

    # テンプレートを縦横比を保ってリサイズ
    th, tw = template.shape[:2]
    scale = min(w / tw, h / th)
    new_w, new_h = int(tw * scale), int(th * scale)
    resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # ポリゴン由来の角度を使う
    angle = snap_to_15(get_angle_from_polygon(pts))

    # 回転
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        resized, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0)
    )

    # 中心合わせで配置
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    x1_new, y1_new = cx - new_w // 2, cy - new_h // 2
    x2_new, y2_new = x1_new + new_w, y1_new + new_h

    # 範囲を画像内に収める
    x1_new, y1_new = max(0, x1_new), max(0, y1_new)
    x2_new, y2_new = min(base_img.shape[1], x2_new), min(base_img.shape[0], y2_new)

    roi = base_img[y1_new:y2_new, x1_new:x2_new]
    if roi.size == 0:
        return base_img

    # マスク処理
    if rotated.shape[2] == 4:  # RGBA
        overlay = rotated[:, :, :3]
        mask = rotated[:, :, 3]
    else:
        overlay = rotated
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # サイズ合わせ
    overlay = cv2.resize(overlay, (x2_new - x1_new, y2_new - y1_new))
    mask = cv2.resize(mask, (x2_new - x1_new, y2_new - y1_new))

    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(overlay, overlay, mask=mask)
    combined = cv2.add(bg, fg)

    base_img[y1_new:y2_new, x1_new:x2_new] = combined
    return base_img

# ======================
# Streamlit アプリ本体
# ======================
st.title("手書き構造図の清書アプリ（OBBの角度利用版）")

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
                    continue

                name = model.names[cls]
                pts = [(int(x), int(y)) for x, y in box.reshape(-1, 2)]

                if name in templates and templates[name] is not None:
                    img = overlay_template(img, templates[name], pts)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="清書結果", use_container_width=True)
    save_path = "output/seisyo_result.png"
    cv2.imwrite(save_path, img)
    st.success(f"清書画像を保存しました: {save_path}")
