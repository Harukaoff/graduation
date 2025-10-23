# all.py
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.linalg import solve

# ============================
# 設定（必要に応じて調整）
# ============================
MODEL_PATH = "runs/obb/train28/weights/best.pt"
TEMPLATE_DIR = "templates"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 材料・断面（仮定）
E = 2.1e11      # ヤング率 (Pa)
I = 8.33e-6     # 慣性モーメント (m^4)
A = 0.01        # 断面積 (m^2)

# ============================
# テンプレート読み込み（透明PNG対応）
# ============================
def load_template(path):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if temp is None:
        return None
    if temp.ndim == 2:
        temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGRA)
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
# 透明PNG合成（安全版）
# ============================
def overlay_template(base_img, template_img, center, angle, scale=1.0):
    if template_img is None:
        return base_img
    # scale clamp to avoid runaway
    scale = float(np.clip(scale, 0.1, 3.0))
    th, tw = template_img.shape[:2]
    new_w, new_h = max(1, int(tw * scale)), max(1, int(th * scale))
    template_resized = cv2.resize(template_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # rotate around center of template
    M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1.0)
    rotated = cv2.warpAffine(template_resized, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    cx, cy = int(center[0]), int(center[1])

    x1 = cx - new_w//2
    y1 = cy - new_h//2
    x2 = x1 + new_w
    y2 = y1 + new_h

    # crop to canvas
    X1 = max(0, x1); Y1 = max(0, y1)
    X2 = min(base_img.shape[1], x2); Y2 = min(base_img.shape[0], y2)
    if X1 >= X2 or Y1 >= Y2:
        return base_img

    tx1 = X1 - x1
    ty1 = Y1 - y1
    tx2 = tx1 + (X2 - X1)
    ty2 = ty1 + (Y2 - Y1)

    roi = base_img[Y1:Y2, X1:X2].astype(float)
    overlay = rotated[ty1:ty2, tx1:tx2]

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3:4].astype(float) / 255.0
        rgb = overlay[:, :, :3].astype(float)
    else:
        alpha = np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=float)
        rgb = overlay.astype(float)

    roi[:] = (1 - alpha) * roi + alpha * rgb
    base_img[Y1:Y2, X1:X2] = roi.astype(np.uint8)
    return base_img

# ============================
# 検出→節点・部材・荷重の抽出
# （ほぼあなたの既存処理を踏襲）
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
        center = (x * results.orig_shape[1], y * results.orig_shape[0]) if hasattr(results, "orig_shape") else (x, y)

        # use pixel coords: result.orig_shape sometimes not set; assume normalized? handle both:
        # If x,y in [0,1] use image dims - but earlier code uses model(img, imgsz=640)[0] which returns x,y in pixels.
        # We'll just treat x,y,w,h as pixels if >1 else normalize by image shape later.

        if name in ["pin", "roller", "fixed"]:
            nodes.append({'name': name, 'pos': center})
        elif name == "beam":
            # treat start/end along local x by w in pixels
            start = (center[0] - w/2, center[1])
            end = (center[0] + w/2, center[1])
            elements.append({'start': start, 'end': end})
        elif name in ["load", "udl", "moment l", "moment r"]:
            loads.append({'name': name, 'pos': center, 'value': 1000.0})
    return nodes, elements, loads

# ============================
# マージ・接続・スナップ（そのまま）
# ============================
def merge_nodes(nodes, threshold=12.0):
    merged = []
    for n in nodes:
        found = False
        for m in merged:
            if np.linalg.norm(np.array(n['pos']) - np.array(m['pos'])) < threshold:
                found = True
                break
        if not found:
            merged.append(n.copy())
    return merged

def connect_beams_to_nodes(elements, nodes):
    connected = []
    for el in elements:
        x1, y1 = el['start']; x2, y2 = el['end']
        start_node = min(nodes, key=lambda n: np.hypot(n['pos'][0]-x1, n['pos'][1]-y1))
        end_node = min(nodes, key=lambda n: np.hypot(n['pos'][0]-x2, n['pos'][1]-y2))
        connected.append({'start': start_node['pos'], 'end': end_node['pos']})
    return connected

def snap_loads_to_beams(loads, elements):
    snapped = []
    for ld in loads:
        lx, ly = ld['pos']
        min_dist = float('inf'); closest = (lx, ly)
        for el in elements:
            x1, y1 = el['start']; x2, y2 = el['end']
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                continue
            t = ((lx - x1)*dx + (ly - y1)*dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            proj_x, proj_y = x1 + t*dx, y1 + t*dy
            dist = np.hypot(lx - proj_x, ly - proj_y)
            if dist < min_dist:
                min_dist = dist
                closest = (proj_x, proj_y)
        snapped.append({'name': ld['name'], 'pos': closest, 'value': ld['value']})
    return snapped

# ============================
# 剛性マトリクス法（各要素の 6x6 を構築して組み上げ）
# nodes: list of {'name','pos'}, elements: list of {'start','end'}, loads: list of {'pos','value'}
# returns: displacement vector u (3*n_nodes)
# ============================
def structural_analysis(nodes, elements, loads, E=E, I=I, A=A):
    n_nodes = len(nodes)
    if n_nodes == 0:
        return np.zeros(0)
    n_dof = 3 * n_nodes
    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)

    # build node index map
    node_map = { tuple(nodes[i]['pos']): i for i in range(n_nodes) }
    node_positions = [tuple(n['pos']) for n in nodes]

    # assemble element stiffness
    for el in elements:
        x1, y1 = el['start']; x2, y2 = el['end']
        L = np.hypot(x2 - x1, y2 - y1)
        if L <= 1e-6:
            continue
        c = (x2 - x1) / L
        s = (y2 - y1) / L

        # local stiffness (axial + bending)
        k_axial = (E * A) / L
        k_b = (E * I) / (L**3)
        k_local = np.array([
            [ k_axial,         0,              0,      -k_axial,       0,           0],
            [ 0,      12*k_b,   6*L*k_b,        0,     -12*k_b,   6*L*k_b],
            [ 0,      6*L*k_b,  4*(L**2)*k_b,   0,     -6*L*k_b,  2*(L**2)*k_b],
            [-k_axial,         0,              0,       k_axial,      0,           0],
            [ 0,     -12*k_b,  -6*L*k_b,        0,      12*k_b,  -6*L*k_b],
            [ 0,      6*L*k_b,  2*(L**2)*k_b,   0,     -6*L*k_b,  4*(L**2)*k_b]
        ], dtype=float)

        # transformation matrix T (6x6)
        T = np.zeros((6,6))
        T[0,0] =  c; T[0,1] = s
        T[1,0] = -s; T[1,1] = c
        T[2,2] = 1
        T[3,3] =  c; T[3,4] = s
        T[4,3] = -s; T[4,4] = c
        T[5,5] = 1

        k_global = T.T @ k_local @ T

        # get dof indices
        # find indices of start and end nodes in node_map
        # due to floating precision, find nearest node
        def find_index_of_point(pt):
            dists = [np.hypot(pt[0]-p[0], pt[1]-p[1]) for p in node_positions]
            return int(np.argmin(dists))
        i = find_index_of_point(el['start'])
        j = find_index_of_point(el['end'])
        dofs = [3*i,3*i+1,3*i+2,3*j,3*j+1,3*j+2]
        for a in range(6):
            for b in range(6):
                K[dofs[a], dofs[b]] += k_global[a,b]

    # assemble loads: assign vertical load to nearest node (simplification)
    for ld in loads:
        lx, ly = ld['pos']
        dists = [np.hypot(lx - p[0], ly - p[1]) for p in node_positions]
        nid = int(np.argmin(dists))
        F[3*nid + 1] += ld['value']  # Fy

    # boundary conditions from node names
    fixed_dofs = []
    for idx, n in enumerate(nodes):
        nm = n['name']
        if nm == "fixed":
            fixed_dofs.extend([3*idx, 3*idx+1, 3*idx+2])
        elif nm == "pin":
            fixed_dofs.extend([3*idx, 3*idx+1])
        elif nm == "roller":
            # roller assumed vertical restraint: fix y -> second DOF
            fixed_dofs.append(3*idx+1)

    free_dofs = np.array([i for i in range(n_dof) if i not in fixed_dofs], dtype=int)

    # handle possible singular K
    if free_dofs.size == 0:
        return np.zeros(n_dof)

    try:
        Kff = K[np.ix_(free_dofs, free_dofs)]
        Ff = F[free_dofs]
        uf = solve(Kff, Ff)
    except Exception as e:
        st.error(f"剛性行列解法でエラー: {e}")
        # return zero displacements to avoid crash
        return np.zeros(n_dof)

    u = np.zeros(n_dof)
    u[free_dofs] = uf
    return u

# ============================
# 応力（変形）図をプロット
# ============================
def plot_stress(nodes, elements, displacements, scale_disp=1.0):
    fig, ax = plt.subplots(figsize=(8,4))
    node_positions = [n['pos'] for n in nodes]

    for el in elements:
        # find node indices
        x1, y1 = el['start']; x2, y2 = el['end']
        idx1 = int(np.argmin([np.hypot(x1 - p[0], y1 - p[1]) for p in node_positions]))
        idx2 = int(np.argmin([np.hypot(x2 - p[0], y2 - p[1]) for p in node_positions]))

        u1 = displacements[3*idx1:3*idx1+2]
        u2 = displacements[3*idx2:3*idx2+2]

        X = [node_positions[idx1][0] + u1[0]*scale_disp, node_positions[idx2][0] + u2[0]*scale_disp]
        Y = [node_positions[idx1][1] + u1[1]*scale_disp, node_positions[idx2][1] + u2[1]*scale_disp]

        ax.plot([node_positions[idx1][0], node_positions[idx2][0]],
                [node_positions[idx1][1], node_positions[idx2][1]], color='k', linewidth=1, alpha=0.4)
        ax.plot(X, Y, color='b', linewidth=3)

        # compute approximate end moments by taking local stiffness * local displacements
        L = np.hypot(node_positions[idx2][0] - node_positions[idx1][0], node_positions[idx2][1] - node_positions[idx1][1])
        if L <= 1e-6:
            continue
        c = (node_positions[idx2][0] - node_positions[idx1][0]) / L
        s = (node_positions[idx2][1] - node_positions[idx1][1]) / L

        # local disp vector [u1 v1 th1 u2 v2 th2]
        th1 = displacements[3*idx1+2]; th2 = displacements[3*idx2+2]
        local_u = np.array([u1[0], u1[1], th1, u2[0], u2[1], th2])

        # rotation matrix T as before
        T = np.zeros((6,6))
        T[0,0] =  c; T[0,1] = s
        T[1,0] = -s; T[1,1] = c
        T[2,2] = 1
        T[3,3] =  c; T[3,4] = s
        T[4,3] = -s; T[4,4] = c
        T[5,5] = 1

        # local displacement = T * global_disp_vector
        # build global vector for these 2 nodes
        global_vec = np.zeros(6)
        global_vec[0:3] = displacements[3*idx1:3*idx1+3]
        global_vec[3:6] = displacements[3*idx2:3*idx2+3]
        local_disp = T @ global_vec

        # compute local k_local (same as in analysis)
        k_axial = (E * A) / L
        k_b = (E * I) / (L**3)
        k_local = np.array([
            [ k_axial,         0,              0,      -k_axial,       0,           0],
            [ 0,      12*k_b,   6*L*k_b,        0,     -12*k_b,   6*L*k_b],
            [ 0,      6*L*k_b,  4*(L**2)*k_b,   0,     -6*L*k_b,  2*(L**2)*k_b],
            [-k_axial,         0,              0,       k_axial,      0,           0],
            [ 0,     -12*k_b,  -6*L*k_b,        0,      12*k_b,  -6*L*k_b],
            [ 0,      6*L*k_b,  2*(L**2)*k_b,   0,     -6*L*k_b,  4*(L**2)*k_b]
        ], dtype=float)

        # internal force local = k_local @ local_disp
        f_local = k_local @ local_disp
        M1 = f_local[2]  # moment at node1 local
        M2 = f_local[5]  # moment at node2 local

        midx = (node_positions[idx1][0] + node_positions[idx2][0]) / 2
        midy = (node_positions[idx1][1] + node_positions[idx2][1]) / 2
        ax.text(midx, midy, f"M1={M1:.1f}\nM2={M2:.1f}", color='red', ha='center', va='bottom', fontsize=9)

    ax.invert_yaxis()
    ax.axis('equal')
    ax.set_title("簡易変形図＋端部モーメント（数値）")
    ax.axis('off')
    return fig

# ============================
# Streamlit アプリ本体
# ============================
def run_app():
    st.title("自動清書 + 剛性マトリクス法解析アプリ")

    # load model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
    else:
        st.warning(f"{MODEL_PATH} が見つかりません。代わりに yolov8n を使用します。")
        model = YOLO("yolov8n.pt")

    conf_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.45, 0.05)
    uploaded_file = st.file_uploader("画像アップロード", type=["jpg","jpeg","png"])

    if uploaded_file is None:
        st.info("画像をアップロードしてください。")
        return

    img_pil = Image.open(uploaded_file).convert("RGB")
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    results = model(img, conf=conf_th, imgsz=640)
    result = results[0]

    # extract nodes/elements/loads
    nodes, elements, loads = extract_nodes_elements(result)
    if len(nodes) == 0 and len(elements) == 0:
        st.warning("要素が検出されませんでした。閾値を下げるか画像を改善してください。")
        st.image(img, caption="入力画像", use_container_width=True)
        return

    # normalize positions: if extracted positions are normalized (0..1) convert to pixels
    # detect whether x looks normalized: if all x < 2 then treat as normalized
    def maybe_denorm(pt):
        x,y = pt
        h,w = img.shape[0], img.shape[1]
        if x <= 1.01 and y <= 1.01:
            return (x * w, y * h)
        else:
            return (x, y)

    # apply denorm to nodes and elements and loads
    for n in nodes:
        n['pos'] = maybe_denorm(n['pos'])
    for el in elements:
        el['start'] = maybe_denorm(el['start'])
        el['end'] = maybe_denorm(el['end'])
    for ld in loads:
        ld['pos'] = maybe_denorm(ld['pos'])

    nodes = merge_nodes(nodes, threshold=12.0)
    elements = connect_beams_to_nodes(elements, nodes)
    loads = snap_loads_to_beams(loads, elements)

    # clear canvas and overlay templates for visualization
    canvas = np.ones_like(img) * 255
    for box in result.obb:
        cls_id = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        if conf < conf_th:
            continue
        name = result.names[cls_id].lower()
        if name not in templates or templates[name] is None:
            continue
        x, y, w, h, angle = box.xywhr.cpu().numpy()[0]
        # x,y might be in pixels or normalized; convert if normalized
        if x <= 1.01 and y <= 1.01:
            cx = int(x * img.shape[1]); cy = int(y * img.shape[0])
            w_px = w * img.shape[1]; h_px = h * img.shape[0]
        else:
            cx = int(x); cy = int(y); w_px = w; h_px = h
        angle_deg = -float(angle * 180.0 / np.pi)
        template = templates[name]
        scale = max(w_px / max(1, template.shape[1]), h_px / max(1, template.shape[0]))
        canvas = overlay_template(canvas, template, (cx, cy), angle_deg, scale)

    # show cleaning result
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="清書結果", use_container_width=True)

    # prepare nodes/elements for analysis (ensure positions as floats)
    # If merge_nodes removed names, we already preserved names
    # Ensure each node has 'name' and 'pos'
    # If there are elements with endpoints not matched to nodes, skip them
    # Build consistent nodes list
    # (merged nodes already have 'name' possibly None)
    for n in nodes:
        if 'name' not in n:
            n['name'] = ''

    if len(nodes) == 0:
        st.error("節点が存在しません。解析できません。")
        return

    # call analysis
    u = structural_analysis(nodes, elements, loads)

    if u.size == 0:
        st.error("解析結果が得られませんでした（自由度が不足または剛性行列が特異）。")
        return

    # plot stress/deformed shape
    fig = plot_stress(nodes, elements, u, scale_disp=1.0)
    st.pyplot(fig)

    # save output
    out_path = os.path.join(OUTPUT_DIR, "result_analysis.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    with open(out_path, "rb") as f:
        st.download_button("解析図をダウンロード", f, file_name="result_analysis.png")

if __name__ == "__main__":
    run_app()
