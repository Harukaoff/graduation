import streamlit as st
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# ==============================
# モデルパス
# ==============================
MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train28\weights\best.pt"


# ==============================
# 節点をスナップ
# ==============================
def snap_nodes(points, threshold=30):
    snapped = []
    for p in points:
        found = False
        for s in snapped:
            if np.linalg.norm(np.array(p) - np.array(s)) < threshold:
                found = True
                break
        if not found:
            snapped.append(p)
    return snapped

# ==============================
# 剛性マトリクス法
# ==============================
def structural_analysis(nodes, elements, loads):
    n = len(nodes)
    if n == 0:
        return np.zeros((1, 1))

    K = np.zeros((2*n, 2*n))
    F = np.zeros((2*n, 1))

    for e in elements:
        i, j = e['node']
        xi, yi = nodes[i]['pos']
        xj, yj = nodes[j]['pos']
        L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        c = (xj-xi)/L
        s = (yj-yi)/L
        k = e.get('E', 20000) * e.get('A', 1) / L
        ke = k * np.array([[ c*c,  c*s, -c*c, -c*s],
                           [ c*s,  s*s, -c*s, -s*s],
                           [-c*c, -c*s,  c*c,  c*s],
                           [-c*s, -s*s,  c*s,  s*s]])
        dof = [2*i, 2*i+1, 2*j, 2*j+1]
        for a in range(4):
            for b in range(4):
                K[dof[a], dof[b]] += ke[a,b]

    for ld in loads:
        nidx = ld['node']
        fx, fy = ld['force']
        F[2*nidx] += fx
        F[2*nidx+1] += fy

    fixed_dofs = []
    for i, nd in enumerate(nodes):
        if nd['type'] in ['pin', 'roller', 'fixed']:
            fixed_dofs.extend([2*i, 2*i+1])
    free_dofs = list(set(range(2*n)) - set(fixed_dofs))

    if len(free_dofs) == 0:
        return np.zeros((2*n, 1))

    Kff = K[np.ix_(free_dofs, free_dofs)]
    Ff = F[free_dofs]
    try:
        uf = np.linalg.solve(Kff, Ff)
    except np.linalg.LinAlgError:
        return np.zeros((2*n, 1))
    u = np.zeros((2*n, 1))
    for idx, dof in enumerate(free_dofs):
        u[dof] = uf[idx]
    return u

# ==============================
# 応力変形図
# ==============================
def plot_stress(nodes, elements, u, scale=200):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_aspect('equal')
    ax.set_title("変形図（赤：変形後）")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    for e in elements:
        i, j = e['node']
        xi, yi = nodes[i]['pos']
        xj, yj = nodes[j]['pos']
        ax.plot([xi, xj], [yi, yj], 'k-', linewidth=1)

    for e in elements:
        i, j = e['node']
        xi, yi = nodes[i]['pos']
        xj, yj = nodes[j]['pos']
        xi_d = xi + u[2*i]*scale
        yi_d = yi + u[2*i+1]*scale
        xj_d = xj + u[2*j]*scale
        yj_d = yj + u[2*j+1]*scale
        ax.plot([xi_d, xj_d], [yi_d, yj_d], 'r--', linewidth=2)

    ax.invert_yaxis()
    return fig

# ==============================
# 曲げモーメント図 & せん断力図
# ==============================
def plot_shear_moment(nodes, elements, loads):
    fig, axs = plt.subplots(2, 1, figsize=(8,8))
    axs[0].set_title("せん断力図 (V)")
    axs[1].set_title("曲げモーメント図 (M)")
    axs[0].set_ylabel("V [kN]")
    axs[1].set_ylabel("M [kN·m]")

    for e in elements:
        i, j = e['node']
        xi, yi = nodes[i]['pos']
        xj, yj = nodes[j]['pos']
        L = np.sqrt((xj-xi)**2 + (yj-yi)**2)
        w = 0
        for ld in loads:
            if ld['type'] in ['UDL']:
                w = 10
        x = np.linspace(0, L, 100)
        V = np.full_like(x, -w*L/2)
        M = w * x * (L - x) / 2
        axs[0].plot(x, V, label=f'Beam {i}-{j}')
        axs[1].plot(x, M, label=f'Beam {i}-{j}')

    axs[0].legend()
    axs[1].legend()
    plt.tight_layout()
    return fig

# ==============================
# Streamlit アプリ
# ==============================
def run_app():
    st.title("構造図自動清書・構造解析・応力図アプリ")

    model = YOLO(MODEL_PATH)
    conf_th = st.slider("検出の信頼度しきい値", 0.0, 1.0, 0.5, 0.05)

    uploaded_file = st.file_uploader("構造図画像をアップロード", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        st.image(img, caption="アップロード画像", use_container_width=True)

        st.info("検出中...")
        results = model(img_np, conf=conf_th)
        result = results[0]

        names = results[0].names
        boxes = results[0].boxes.xywh.cpu().numpy() if results[0].boxes is not None else []
        st.write("Detected class names:", names)
        st.write("Detected boxes:", boxes)

        if result.boxes is None or len(result.boxes) == 0:
            st.error("要素が検出されませんでした。")
            return

        boxes = result.boxes.xywh.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = [result.names[c] for c in classes]

        nodes, elements, loads = [], [], []
        node_points = []

        for (x, y, w, h), name in zip(boxes, names):
            if name in ["pin", "roller", "fixed"]:
                nodes.append({"pos": (x, y), "type": name})
                node_points.append((x, y))
            elif name == "beam":
                elements.append({"node": [], "E": 20000, "A": 1, "center": (x, y), "size": (w, h)})
                node_points.extend([(x - w/2, y), (x + w/2, y)])
            elif name in ["load", "UDL", "moment l", "moment r"]:
                loads.append({"pos": (x, y), "type": name})

        snapped = snap_nodes(node_points)

        for e in elements:
            x, y = e["center"]
            w, h = e["size"]
            ends = [(x - w/2, y), (x + w/2, y)]
            e["node"] = [np.argmin([np.linalg.norm(np.array(pt)-np.array(s)) for s in snapped]) for pt in ends]

        for ld in loads:
            nidx = np.argmin([np.linalg.norm(np.array(ld["pos"]) - np.array(s)) for s in snapped])
            if ld["type"] == "load":
                loads[loads.index(ld)] = {"node": nidx, "force": (0, -1000), "type": "load"}
            elif ld["type"] == "UDL":
                loads[loads.index(ld)] = {"node": nidx, "force": (0, -500), "type": "UDL"}
            else:
                loads[loads.index(ld)] = {"node": nidx, "force": (0, 0), "type": ld["type"]}

        nodes = [{"pos": s, "type": next((n["type"] for n in nodes if np.linalg.norm(np.array(n["pos"])-np.array(s))<30), "free")} for s in snapped]

        if len(elements) == 0:
            st.error("梁が検出されませんでした。")
            return

        u = structural_analysis(nodes, elements, loads)
        fig_def = plot_stress(nodes, elements, u)
        st.pyplot(fig_def)

        fig_sm = plot_shear_moment(nodes, elements, loads)
        st.pyplot(fig_sm)

if __name__ == "__main__":
    run_app()
