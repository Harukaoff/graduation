import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ============================
# 設定
# ============================
MODEL_PATH = "runs/obb/train28/weights/best.pt"
TEMPLATE_DIR = "templates"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# テンプレート読み込み
# ============================
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

# ============================
# テンプレート貼り付け
# ============================
def overlay_template(base_img, template_img, center, angle, scale=1.0):
    h, w = template_img.shape[:2]
    new_w, new_h = max(int(w * scale),1), max(int(h * scale),1)
    template_resized = cv2.resize(template_img, (new_w, new_h))
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
    rotated = cv2.warpAffine(template_resized, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    x, y = int(center[0]), int(center[1])
    x1, y1 = max(x - new_w // 2, 0), max(y - new_h // 2, 0)
    x2, y2 = min(x1 + new_w, base_img.shape[1]), min(y1 + new_h, base_img.shape[0])
    roi = base_img[y1:y2, x1:x2]
    rot_crop = rotated[0:(y2 - y1), 0:(x2 - x1)]
    if rot_crop.shape[2] == 4:
        alpha = rot_crop[:, :, 3:] / 255.0
        roi = (1 - alpha) * roi + alpha * rot_crop[:, :, :3]
    base_img[y1:y2, x1:x2] = roi.astype(np.uint8)
    return base_img

# ============================
# 節点・梁・荷重抽出
# ============================
def extract_nodes_elements(results):
    nodes = []
    elements = []
    loads = []

    if not hasattr(results, "obb") or results.obb is None:
        return nodes, elements, loads

    for box in results.obb:
        cls_id = int(box.cls.cpu().numpy()[0])
        name = results.names[cls_id]
        x, y, w, h, angle = box.xywhr.cpu().numpy()[0]
        angle_deg = -angle * 180 / np.pi
        center = (x, y)

        if name in ["pin", "roller", "fixed"]:
            nodes.append({'name': name, 'pos': center})
        elif name == "beam":
            # 簡易的に水平梁と仮定
            start = (x - w / 2, y)
            end = (x + w / 2, y)
            elements.append({'start': start, 'end': end})
        elif name == "load":
            loads.append({'pos': center, 'value': 1000})  # 仮の荷重値

    return nodes, elements, loads

# ============================
# 節点統合
# ============================
def merge_nodes(nodes, threshold=10):
    merged = []
    for n in nodes:
        found = False
        for m in merged:
            if np.linalg.norm(np.array(n['pos']) - np.array(m['pos'])) < threshold:
                found = True
                break
        if not found:
            merged.append(n)
    return merged

# ============================
# 梁端点を節点に接続
# ============================
def connect_beams_to_nodes(elements, nodes):
    connected_elements = []
    for el in elements:
        x1, y1 = el['start']
        x2, y2 = el['end']
        start_node = min(nodes, key=lambda n: np.linalg.norm(np.array(n['pos']) - np.array([x1,y1])))
        end_node = min(nodes, key=lambda n: np.linalg.norm(np.array(n['pos']) - np.array([x2,y2])))
        connected_elements.append({'start': start_node['pos'], 'end': end_node['pos']})
    return connected_elements

# ============================
# 簡易剛性マトリクス法解析（表示のみ）
# ============================
def simple_structural_analysis(nodes, elements, loads):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_title("簡易応力図（接続済み）")
    for el in elements:
        x1, y1 = el['start']
        x2, y2 = el['end']
        ax.plot([x1, x2], [y1, y2], 'k', lw=3)
    for ld in loads:
        x, y = ld['pos']
        ax.arrow(x, y, 0, -20, head_width=5, head_length=10, fc='r', ec='r')
    for nd in nodes:
        x, y = nd['pos']
        ax.plot(x, y, 'bo')
        ax.text(x+2, y+2, nd['name'], color='b')
    ax.invert_yaxis()
    ax.axis('off')
    return fig

# ============================
# Streamlit アプリ
# ============================
def run_app():
    st.title("手書き構造図自動清書＆接続解析")

    # モデルロード
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
    else:
        st.warning(f"{MODEL_PATH} が見つかりません。公式モデルを使用")
        model = YOLO("yolov8n.pt")

    conf_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.05)

    uploaded_file = st.file_uploader("画像アップロード", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        results = model(img, conf=conf_th, imgsz=640)[0]

        canvas = np.ones_like(img) * 255
        nodes, elements, loads = extract_nodes_elements(results)

        # 節点統合
        nodes = merge_nodes(nodes)
        # 梁端点を節点に接続
        elements = connect_beams_to_nodes(elements, nodes)

        # 清書
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
                angle_deg = -angle * 180 / np.pi
                template = templates[name]
                scale = max(w / template.shape[1], 0.1)
                canvas = overlay_template(canvas, template, (x,y), angle_deg, scale)
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="清書結果", use_container_width=True)
        out_path = os.path.join(OUTPUT_DIR, "output_overlay.png")
        Image.fromarray(img_rgb).save(out_path)
        with open(out_path,"rb") as f:
            st.download_button("結果をダウンロード", f, file_name="output_overlay.png")

        # 簡易解析
        st.subheader("簡易構造解析")
        fig = simple_structural_analysis(nodes, elements, loads)
        st.pyplot(fig)

if __name__ == "__main__":
    run_app()
