import os
import numpy as np
import matplotlib.pyplot as plt

def load_labels(label_file):
    """ YOLO OBB形式のラベルを読み込む """
    elems = []
    with open(label_file, "r") as f:
        for line in f:
            cls, cx, cy, w, h, angle = map(float, line.strip().split())
            elems.append((cls, cx, cy, w, h, angle))
    return elems

def obb_to_nodes(cx, cy, w, h, angle):
    """ OBBを両端点に変換 """
    rad = np.deg2rad(angle)
    dx = (w/2) * np.cos(rad)
    dy = (w/2) * np.sin(rad)
    x1, y1 = cx - dx, cy - dy
    x2, y2 = cx + dx, cy + dy
    return (x1, y1), (x2, y2)

def snap_point(p, nodes, tol=10):
    """ 既存ノードに近ければそれを使う """
    for i, q in enumerate(nodes):
        if np.linalg.norm(np.array(p) - np.array(q)) < tol:
            return i
    nodes.append(p)
    return len(nodes)-1

def build_structure(labels, snap_tol=10):
    nodes, elements = [], []
    for _, cx, cy, w, h, angle in labels:
        p1, p2 = obb_to_nodes(cx, cy, w, h, angle)
        i = snap_point(p1, nodes, snap_tol)
        j = snap_point(p2, nodes, snap_tol)
        elements.append((i, j))
    return nodes, elements

def draw_structure(nodes, elements, out_file):
    plt.figure(figsize=(6,6))
    for i, (x,y) in enumerate(nodes):
        plt.plot(x, y, "ro")
        plt.text(x+2, y+2, str(i), fontsize=8)
    for (i,j) in elements:
        x1,y1 = nodes[i]
        x2,y2 = nodes[j]
        plt.plot([x1,x2], [y1,y2], "b-")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.savefig(out_file)
    plt.close()

if __name__ == "__main__":
    labels = load_labels("dataset/train/labels/sample.txt")
    nodes, elements = build_structure(labels, snap_tol=10)
    draw_structure(nodes, elements, "output/structure.png")
    print("節点:", nodes)
    print("要素:", elements)
