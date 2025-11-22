import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os, math

# -----------------------
# 基本ツール
# -----------------------
def load_template(path):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if temp is None:
        return None
    if temp.shape[2] == 3:
        b, g, r = cv2.split(temp)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        temp = cv2.merge([b, g, r, alpha])
    return temp

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

def rotate_vec(vec, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    x, y = vec
    return (x * ca - y * sa, x * sa + y * ca)

def overlay_template(base_img, template_img, center, angle_deg, scale=1.0):
    if template_img is None:
        return base_img
    h, w = template_img.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    template_resized = cv2.resize(template_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(template_resized, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    x, y = int(center[0]), int(center[1])
    x1, y1 = max(0, x - new_w // 2), max(0, y - new_h // 2)
    x2, y2 = min(x1 + new_w, base_img.shape[1]), min(y1 + new_h, base_img.shape[0])
    if y2 <= y1 or x2 <= x1:
        return base_img
    rot_crop = rotated[0:(y2 - y1), 0:(x2 - x1)]
    roi = base_img[y1:y2, x1:x2]
    alpha = rot_crop[:, :, 3] / 255.0
    for c in range(3):
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * rot_crop[:, :, c]
    base_img[y1:y2, x1:x2] = roi
    return base_img

# -----------------------
# アンカー計算
# -----------------------
def compute_anchors(name, center, angle_deg, scale, template_img):
    anchors = []
    th, tw = template_img.shape[:2]
    if name == "beam":
        half = (tw * scale) / 2.0
        lt = (-half, 0)
        rt = (half, 0)
        anchors = [
            (center[0] + rotate_vec(lt, angle_deg)[0], center[1] + rotate_vec(lt, angle_deg)[1]),
            (center[0] + rotate_vec(rt, angle_deg)[0], center[1] + rotate_vec(rt, angle_deg)[1])
        ]
    elif name in ("pin", "roller", "fixed"):
        off = (0, (th * scale) / 2.0)
        ax, ay = rotate_vec(off, angle_deg)
        anchors = [(center[0] + ax, center[1] + ay)]
    else:
        anchors = [(center[0], center[1])]
    return anchors

# -----------------------
# 荷重作用点の計算
# -----------------------
def compute_action_point(name, center, angle_deg, scale, template_img):
    th, tw = template_img.shape[:2]
    if name == "load":
        # 矢印の先端を想定: 上から下向き矢印なら、中心から"上側"に半分動く
        offset = (0, -th * scale / 2)
    elif name == "udl":
        # 分布荷重は中心付近に取る
        offset = (0, -th * scale / 4)
    elif name in ("moment_l", "moment_r"):
        # モーメント記号の対角線の中点
        offset = (0, 0)
    else:
        return None
    ox, oy = rotate_vec(offset, angle_deg)
    return (center[0] + ox, center[1] + oy)

# -----------------------
# Streamlit main
# -----------------------
def run_app():
    st.title("構造図 清書（荷重作用点付き）")
    model = YOLO("runs/obb/train28/weights/best.pt")

    conf_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.05)
    uploaded_file = st.file_uploader("構造図画像をアップロードしてください", type=["jpg","jpeg","png"])
    if uploaded_file is None:
        return

    img_pil = Image.open(uploaded_file).convert("RGB")
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    results = model(img, conf=conf_th, imgsz=640)[0]
    canvas = np.ones_like(img) * 255

    if not hasattr(results, "obb") or results.obb is None:
        st.warning("検出結果がありません")
        return

    for box in results.obb:
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        if conf < conf_th:
            continue
        name = results.names[cls_id]
        if name not in templates or templates[name] is None:
            continue
        x, y, w, h, angle = box.xywhr.cpu().numpy()[0]
        angle_deg = -float(angle * 180.0 / math.pi)
        template = templates[name]
        scale = max(0.01, w / max(1.0, template.shape[1]))

        # 描画
        canvas = overlay_template(canvas, template, (x, y), angle_deg, scale)
        cv2.putText(canvas, f"{name} ({conf:.2f})", (int(x)+5, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        # 荷重系の場合：作用点を赤で表示
        if name in ("load", "udl", "moment_l", "moment_r"):
            act = compute_action_point(name, (x, y), angle_deg, scale, template)
            if act is not None:
                cv2.circle(canvas, (int(act[0]), int(act[1])), 6, (0,0,255), -1)
                cv2.putText(canvas, "作用点", (int(act[0])+6, int(act[1])-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
             caption="白紙上の清書（荷重作用点付き）",
             use_container_width=True)

if __name__ == "__main__":
    run_app()
