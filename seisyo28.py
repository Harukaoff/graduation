import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(layout="wide", page_title="構造図 清書 (テンプレ最上点節点/梁端接続/荷重表示)")

# ==== 設定 ====
MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
TEMPLATE_DIR = r"C:\Users\morim\Downloads\graduation\templates"
TEMPLATE_FILES = {
    "pin": "pin.png",
    "roller": "roller.png",
    "fixed": "fixed.png",
    "beam": "beam.png",
    "load": "load.png",
    "momentl": "momentL.png",
    "momentr": "momentR.png",
    "udl": "UDL.png",
    "hinge": "hinge.png",
}
support_types = {"pin","roller","fixed","hinge"}
load_types = {"load","udl","momentl","momentr"}

def template_path(name):
    fname = TEMPLATE_FILES.get(name)
    return os.path.join(TEMPLATE_DIR, fname) if fname else None

def to_numpy(x):
    try: return x.cpu().numpy()
    except Exception: return np.array(x)

def order_cw_start_top_left(pts):
    pts = np.asarray(pts, float).reshape(-1,2)
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    angles = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
    order = np.argsort(-angles)
    pts_sorted = pts[order]
    miny = np.min(pts_sorted[:,1])
    cand = np.where(np.isclose(pts_sorted[:,1], miny, atol=1e-2))[0]
    idx = cand[np.argmin(pts_sorted[cand,0])] if len(cand)>1 else cand[0]
    pts_final = np.roll(pts_sorted, -idx, axis=0)
    return pts_final

def load_template_rgba(path):
    if not path or not os.path.exists(path): return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        b,g,r = cv2.split(img)
        a = np.ones_like(b)*255
        img = cv2.merge([b,g,r,a])
    return img

def scale_image(img, scale):
    h,w = img.shape[:2]
    nw = max(1, int(w*scale))
    nh = max(1, int(h*scale))
    return cv2.resize(img, (nw,nh), interpolation=cv2.INTER_AREA)

def rotate_image_keep_alpha(img, angle_deg):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle_deg, 1.0)
    abs_cos = abs(M[0,0])
    abs_sin = abs(M[0,1])
    nw = int(h*abs_sin + w*abs_cos)
    nh = int(h*abs_cos + w*abs_sin)
    M[0,2] += (nw/2 - w/2)
    M[1,2] += (nh/2 - h/2)
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def overlay_rgba(base, overlay, center):
    bx, by = int(center[0]), int(center[1])
    oh, ow = overlay.shape[:2]
    x1 = bx - ow//2; y1 = by - oh//2
    X1 = max(0,x1); X2 = min(base.shape[1], x1+ow)
    Y1 = max(0,y1); Y2 = min(base.shape[0], y1+oh)
    ox1 = X1-x1; oy1 = Y1-y1
    ox2 = ox1 + (X2-X1); oy2 = oy1 + (Y2-Y1)
    crop = overlay[oy1:oy2, ox1:ox2]
    if crop.shape[2] < 4:
        base[Y1:Y2, X1:X2] = crop[..., :3]
        return base
    alpha = crop[..., 3:4]/255.0
    for c in range(3):
        base[Y1:Y2, X1:X2, c] = (1.0-alpha[...,0])*base[Y1:Y2, X1:X2, c] + alpha[...,0]*crop[...,c]
    return base

def get_template_top_point(tpl):
    # テンプレ画像の最上部の画素中心（α>128）。複数ならx最小の点
    assert tpl is not None
    alpha = tpl[...,3]
    pts = np.column_stack(np.where(alpha>128))
    if len(pts)==0:
        h,w = tpl.shape[:2]
        return np.array([w//2,0])
    miny = np.min(pts[:,0])
    cand = pts[np.where(pts[:,0]==miny)]
    minx = np.min(cand[:,1])
    top_pt = np.array([minx, miny]) # (x,y) in template coord
    return top_pt

def template_absolute_top(img_abs_center, template, angle=0):
    # テンプレ画像で回転も考慮し、最上端が本体絶対座標でどこか計算
    h,w = template.shape[:2]
    top_pt = get_template_top_point(template)
    offset = top_pt - np.array([w//2, h//2])
    # rotate offset
    theta = np.deg2rad(angle)
    rotM = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    rotated_offset = rotM @ offset
    return img_abs_center + rotated_offset

def align_nodes_y(nodes, thresh=8.0):
    ys = np.array([n[1] for n in nodes])
    used = np.zeros(len(nodes), dtype=bool)
    new_nodes = list(nodes)
    for i in range(len(nodes)):
        if used[i]: continue
        group = [i]
        for j in range(i+1, len(nodes)):
            if abs(ys[i]-ys[j]) < thresh: group.append(j)
        if len(group)>1:
            avg_y = np.mean([ys[g] for g in group])
            for g in group:
                new_nodes[g] = np.array([new_nodes[g][0], avg_y])
                used[g] = True
        else:
            used[group[0]] = True
    return new_nodes

def get_beam_endpoints(pts):
    edges = []
    for i in range(4):
        e = {'i':i,'p1':pts[i],'p2':pts[(i+1)%4],'vec':pts[(i+1)%4]-pts[i]}
        e['len'] = np.linalg.norm(e['vec'])
        edges.append(e)
    long_idx = np.argsort([e['len'] for e in edges])[-2:]
    P = [(edges[i]["p1"],edges[i]["p2"]) for i in long_idx]
    # 最も離れた2点
    dmax, pt1, pt2 = -1, None, None
    for p1 in pts:
        for p2 in pts:
            d = np.linalg.norm(p1-p2)
            if d > dmax:
                dmax = d; pt1 = p1; pt2 = p2
    return pt1, pt2

def round_angle_deg(angle):
    # 15度刻み
    return round(angle/15)*15

def find_nearest_node(pt, nodes):
    dists = [np.linalg.norm(pt-n) for n in nodes]
    return int(np.argmin(dists)) if len(nodes)>0 else -1

st.title("🦾 構造図 清書（テンプレ最上点節点/梁端接続/荷重表示）")
st.write("- ピン・ピンローラー・ヒンジはテンプレート画像の最上部画素位置を節点（ノード）座標とします")
st.write("- 梁と支点は、梁の両端 (四角形一番離れた2点) と支点節点の近傍(snap)で接続。コの字型もOK")
st.write("- 荷重: 'load','udl','moment'を梁上の最近傍点に配置")

conf_th = st.slider("検出信頼度", 0.2, 1.0, 0.45, 0.01)
y_align_th = st.slider("高さ揃え閾値(px)", 2.0, 32.0, 8.0, 1.0)
node_connect_th = st.slider("部材端と支点節点の近接閾値 (px)", 10, 50, 25, 1)
uploaded = st.file_uploader("構造図画像アップロード", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("画像ファイルをアップロードしてください。")
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
st.image(img_pil, caption="元画像", use_container_width=True)
TEMPL = {k: load_template_rgba(template_path(k)) for k in TEMPLATE_FILES}
if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    st.error(f"モデルパスが存在しません: {MODEL_PATH}")
    st.stop()
if not st.button("実行（清書・接続・節点荷重表示）"):
    st.stop()

with st.spinner("推論実行中..."):
    model = YOLO(MODEL_PATH)
    res = model(img, conf=conf_th, imgsz=640)[0]
    obb = res.obb

supports, beams, loads = [], [], []
N = len(to_numpy(obb.xyxyxyxy)) if hasattr(obb, "xyxyxyxy") else 0
for i in range(N):
    conf = float(to_numpy(obb.conf[i]))
    if conf < conf_th: continue
    cls_id = int(to_numpy(obb.cls[i]))
    name = res.names[cls_id].lower().replace(" ", "")
    pts = to_numpy(obb.xyxyxyxy[i]).reshape(4,2)
    pts = order_cw_start_top_left(pts)
    angle = round_angle_deg(
        math.degrees(math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0])) if name!="beam" else
        math.degrees(math.atan2(pts[2][1]-pts[0][1], pts[2][0]-pts[0][0]))
    )
    if name in support_types:
        tpl = TEMPL.get(name)
        node = None
        if tpl is not None:
            node = template_absolute_top(pts.mean(axis=0), tpl, angle)
        else:
            node = pts.mean(axis=0)
        supports.append(dict(type=name, node=node, pts=pts, angle=angle, conf=conf))
    elif name=="beam":
        beams.append({"type": "beam", "pts": pts, "angle": round_angle_deg(angle), "conf": conf})
    elif name in load_types:
        loads.append({"type": name, "pts": pts, "angle": round_angle_deg(angle), "conf": conf})

nodes = np.array([s["node"] for s in supports]) if supports else np.empty((0,2))
nodes = align_nodes_y(nodes, thresh=y_align_th) if len(nodes)>=2 else nodes
for i,s in enumerate(supports): s["node"] = nodes[i]

# 梁端接続: 梁端 (get_beam_endpoints) をそれぞれ支点節点に近接スナップ
beam_connections = []
for b in beams:
    pt1, pt2 = get_beam_endpoints(b['pts'])
    idx1 = find_nearest_node(pt1, nodes)
    idx2 = find_nearest_node(pt2, nodes)
    n1, n2 = nodes[idx1], nodes[idx2]
    # 梁端点(支点が近ければスナップ)＝正確なコの字もOK
    if np.linalg.norm(pt1-n1)<node_connect_th: pt1 = n1
    if np.linalg.norm(pt2-n2)<node_connect_th: pt2 = n2
    beam_connections.append({
        "nidx_a": int(idx1),
        "nidx_b": int(idx2),
        "a": pt1.tolist(),
        "b": pt2.tolist(),
        "angle": b["angle"],
        "conf": float(b["conf"])
    })

# 荷重: 梁線分ごとに最近傍投影
load_connections = []
for l in loads:
    c = l["pts"].mean(axis=0)
    best_beam = None
    best_proj = None
    best_dist = 1e9
    for beam in beam_connections:
        a = np.array(beam["a"])
        b = np.array(beam["b"])
        ba = b-a
        denom = np.dot(ba,ba)+1e-12
        t = np.dot(c-a, ba)/denom
        t = max(0.0, min(1.0, t))
        proj = a + t*ba
        dist = np.linalg.norm(c-proj)
        if dist<best_dist:
            best_dist = dist
            best_beam = beam
            best_proj = proj
    load_connections.append({
        "type": l["type"],
        "on_beam": {"nidx_a": best_beam["nidx_a"], "nidx_b": best_beam["nidx_b"]} if best_beam else None,
        "proj_pt": best_proj.tolist() if best_proj is not None else c.tolist(),
        "angle": l["angle"],
        "conf": float(l["conf"])
    })

# 清書 & ビジュアライズ
cleaned = np.ones_like(img)*255
# 梁線分
for conn in beam_connections:
    pt1, pt2 = np.array(conn["a"]), np.array(conn["b"])
    cv2.line(cleaned, tuple(map(int,pt1)), tuple(map(int,pt2)), (160,160,160), 6)
    # 梁テンプレ中心
    tplb = TEMPL.get("beam")
    if tplb is not None:
        center = (pt1+pt2)/2
        factor = np.linalg.norm(pt2-pt1)/max(tplb.shape[:2])
        tplb_scaled = scale_image(tplb, factor)
        tplb_rot    = rotate_image_keep_alpha(tplb_scaled, conn["angle"])
        cleaned = overlay_rgba(cleaned, tplb_rot, center)
# 支点テンプレ貼り＋節点
for i,s in enumerate(supports):
    name = s["type"]
    tpl = TEMPL.get(name)
    center = s["node"]
    angle = s["angle"]
    if tpl is not None:
        tpl_scaled = scale_image(tpl, 0.8)
        tpl_rot = rotate_image_keep_alpha(tpl_scaled, angle)
        cleaned = overlay_rgba(cleaned, tpl_rot, center)
    cv2.circle(cleaned, tuple(map(int,center)), 10, (0,0,255), 2)
    cv2.putText(cleaned, f"N{i}", (int(center[0])+8, int(center[1])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
# 荷重テンプレ（梁最近傍に描画）
for l in load_connections:
    name = l["type"]
    tpl = TEMPL.get(name)
    center = l["proj_pt"]
    angle = l["angle"]
    if tpl is not None:
        tpl_scaled = scale_image(tpl, 0.9)
        tpl_rot    = rotate_image_keep_alpha(tpl_scaled, angle)
        cleaned = overlay_rgba(cleaned, tpl_rot, center)
    cv2.circle(cleaned, tuple(map(int,center)), 8, (0,128,255), 2)

st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB),"清書画像（テンプレ節点/梁端接続/荷重）",use_container_width=True)
st.subheader("節点テーブル")
st.table([
    {
        "id":i,
        "type": s["type"],
        "x": float(s["node"][0]),
        "y": float(s["node"][1]),
        "conf": float(s["conf"])
    }
    for i,s in enumerate(supports)
])
st.subheader("梁接続テーブル")
st.table(beam_connections)
st.subheader("荷重投影テーブル")
st.table(load_connections)
st.success("テンプレ最上部節点・梁端支点接続・荷重可視化（コの字対応） 完了！")