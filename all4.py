import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.linalg import solve

# ============================
# 設定
# ============================
MODEL_PATH = "runs/obb/train28/weights/best.pt"
TEMPLATE_DIR = "templates"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

E = 2.1e11      # ヤング率 (Pa)
I = 8.33e-6     # 慣性モーメント (m^4)
A = 0.01        # 梁断面積 (m^2)

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
# 節点・梁・荷重抽出（荷重4種のみ）
# ============================
def extract_nodes_elements(results):
    nodes = []
    elements = []
    loads = []

    if not hasattr(results, "obb") or results.obb is None:
        return nodes, elements, loads

    for box in results.obb:
        cls_id = int(box.cls.cpu().numpy()[0])
        name = results.names[cls_id].lower()
        x, y, w, h, angle = box.xywhr.cpu().numpy()[0]
        center = (x, y)

        if name in ["pin", "roller", "fixed"]:
            nodes.append({'name': name, 'pos': center})
        elif name == "beam":
            start = (x - w / 2, y)
            end = (x + w / 2, y)
            elements.append({'start': start, 'end': end})
        elif name in ["load", "udl", "moment l", "moment r"]:
            loads.append({'name': name, 'pos': center, 'value': 1000})  # 仮の荷重値

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
# 荷重を最寄り梁上にスナップ
# ============================
def snap_loads_to_beams(loads, elements):
    snapped_loads = []
    for ld in loads:
        lx, ly = ld['pos']
        min_dist = float('inf')
        closest_point = (lx, ly)
        for el in elements:
            x1, y1 = el['start']
            x2, y2 = el['end']
            dx, dy = x2 - x1, y2 - y1
            if dx == dy == 0:
                continue
            t = ((lx - x1) * dx + (ly - y1) * dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            proj_x, proj_y = x1 + t*dx, y1 + t*dy
            dist = np.hypot(lx - proj_x, ly - proj_y)
            if dist < min_dist:
                min_dist = dist
                closest_point = (proj_x, proj_y)
        snapped_loads.append({'name': ld['name'], 'pos': closest_point, 'value': ld['value']})
    return snapped_loads

# ============================
# 簡易2D梁剛性マトリクス法
# ============================
def structural_analysis(nodes, elements, loads):
    n_dof = 3 * len(nodes)  # u_x, u_y, theta_z
    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)

    node_map = {tuple(n['pos']): i for i,n in enumerate(nodes)}

    for el in elements:
        x1, y1 = el['start']
        x2, y2 = el['end']
        L = np.hypot(x2 - x1, y2 - y1)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        k_local = (E*I / L**3) * np.array([
            [ A*L**2/I, 0, 0, -A*L**2/I, 0, 0],
            [0, 12, 6*L, 0, -12, 6*L],
            [0, 6*L, 4*L**2, 0, -6*L, 2*L**2],
            [-A*L**2/I, 0, 0, A*L**2/I, 0, 0],
            [0, -12, -6*L, 0, 12, -6*L],
            [0, 6*L, 2*L**2, 0, -6*L, 4*L**2]
        ])
        dofs = []
        for pt in [el['start'], el['end']]:
            idx = node_map[tuple(pt)]
            dofs.extend([3*idx, 3*idx+1, 3*idx+2])
        for i in range(6):
            for j in range(6):
                K[dofs[i], dofs[j]] += k_local[i,j]

    for ld in loads:
        idx = min(node_map.keys(), key=lambda n: np.hypot(n[0]-ld['pos'][0], n[1]-ld['pos'][1]))
        node_idx = node_map[idx]
        F[3*node_idx+1] += ld['value']

    # 支点拘束
    fixed_dofs = []
    for i,n in enumerate(nodes):
        if n['name'] == "fixed":
            fixed_dofs.extend([3*i, 3*i+1, 3*i+2])
        elif n['name'] == "pin":
            fixed_dofs.extend([3*i, 3*i+1])
        elif n['name'] == "roller":
            fixed_dofs.append(3*i+1)
    free_dofs = np.array([i for i in range(n_dof) if i not in fixed_dofs])
    u = np.zeros(n_dof)
    u[free_dofs] = solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])
    return u

# ============================
# 応力図表示
# ============================
def plot_stress(nodes, elements, displacements):
    fig, ax = plt.subplots(figsize=(8,4))
    for el in elements:
        n1, n2 = el['start'], el['end']
        idx1, idx2 = [i for i,n in enumerate(nodes) if tuple(n['pos'])==tuple(pt)] 
        u1 = displacements[3*idx1:3*idx1+2]
        u2 = displacements[3*idx2:3*idx2+2]
        x = [n1[0]+u1[0], n2[0]+u2[0]]
        y = [n1[1]+u1[1], n2[1]+u2[1]]
        ax.plot(x, y, 'b', lw=3)
    ax.invert_yaxis()
    ax.set_title("簡易応力変形図")
    ax.axis('off')
    return fig

# ============================
# Streamlit アプリ
# ============================
def run_app():
    st.title("自動清書 + 剛性マトリクス法解析アプリ")
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
        nodes = merge_nodes(nodes)
        elements = connect_beams_to_nodes(elements, nodes)
        loads = snap_loads_to_beams(loads, elements)

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

        st.subheader("剛性マトリクス法解析")
        u = structural_analysis(nodes, elements, loads)
        fig = plot_stress(nodes, elements, u)
        st.pyplot(fig)

if __name__ == "__main__":
    run_app()
