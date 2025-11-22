import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(layout="wide", page_title="構造図 清書 (支点接続/荷重節点/15度刻み)")

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

def top_vertex_of_support(pts):
    ys = pts[:,1]
    miny = ys.min()
    y_cands = np.where(np.isclose(ys, miny, atol=1e-3))[0]
    xs = pts[y_cands,0]
    idx = y_cands[np.argmin(xs)]
    return pts[idx]

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

def edge_list_from_pts(pts):
    edges = []
    for i in range(4):
        p1 = pts[i]; p2 = pts[(i+1)%4]
        vec = p2-p1
        length = np.linalg.norm(vec)
        angle = math.degrees(math.atan2(vec[1], vec[0]))
        mid = (p1+p2)/2.0
        edges.append({"i":i, "p1":p1, "p2":p2, "vec":vec, "len":length, "angle":angle, "mid":mid})
    return edges

def index_of_short_edges(edges):
    lens = [e["len"] for e in edges]
    idcs = np.argsort(lens)
    return [int(idcs[0]), int(idcs[1])]

def index_of_long_edges(edges):
    lens = [e["len"] for e in edges]
    idcs = np.argsort(lens)
    return [int(idcs[-1]), int(idcs[-2])]

def find_nearest_node(pt, nodes):
    dists = [np.linalg.norm(pt-n) for n in nodes]
    return int(np.argmin(dists)) if len(dists)>0 else -1

def round_angle_deg(angle):
    # 15度刻み
    return round(angle/15)*15

def get_beam_angle_and_unit(pts):
    # 四角形梁検出 ptsから、長辺のベクトルと角度
    edges = edge_list_from_pts(pts)
    long_idx = index_of_long_edges(edges)
    e = edges[long_idx[0]]
    dx,dy = e['vec'][0],e['vec'][1]
    angle = math.degrees(math.atan2(dy, dx))
    unit = (e['vec'])/(np.linalg.norm(e['vec'])+1e-12)
    return angle, unit, e['p1'], e['p2']

st.title("🦾 構造図 清書（支点結合で梁長さ調整/節点/荷重部材・15度刻み）")
st.write("- ピン/ローラー/ヒンジも節点")
st.write("- 梁端は検出された支点（節点）を結ぶ形で長さとし角度は検出どおり")
st.write("- 梁・荷重の角度は15度刻みに強制")
st.write("- 荷重に関する要素は梁上に配置（closest projection）")

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

# 支点とヒンジ
support_types = {"pin","roller","fixed","hinge"}
others_types = {"load","udl","momentl","momentr"}
supports, beams, loads = [], [], []
N = len(to_numpy(obb.xyxyxyxy)) if hasattr(obb, "xyxyxyxy") else 0
for i in range(N):
    conf = float(to_numpy(obb.conf[i]))
    if conf < conf_th: continue
    cls_id = int(to_numpy(obb.cls[i]))
    name = res.names[cls_id].lower().replace(" ", "")
    pts = to_numpy(obb.xyxyxyxy[i]).reshape(4,2)
    pts = order_cw_start_top_left(pts)
    if name in support_types:
        node = top_vertex_of_support(pts) if name in ("pin", "roller","hinge") else pts.mean(axis=0)
        supports.append(dict(type=name, node=node, pts=pts, conf=conf))
    elif "beam" in name:
        beams.append({"type": "beam", "pts": pts, "conf": conf})
    elif name in others_types:
        loads.append({"type": name, "pts": pts, "conf": conf})

# 節点高さ揃え
nodes = np.array([s["node"] for s in supports]) if supports else np.empty((0,2))
nodes = align_nodes_y(nodes, thresh=y_align_th) if len(nodes)>=2 else nodes
for i,s in enumerate(supports): s["node"] = nodes[i]

# 梁端点 節点近接スナップ
beam_connections = []
for b in beams:
    angle_deg, unitvec, p1, p2 = get_beam_angle_and_unit(b['pts'])
    # 最も近い節点を両端として使う
    idx1 = find_nearest_node(p1, nodes)
    idx2 = find_nearest_node(p2, nodes)
    node_a, node_b = nodes[idx1], nodes[idx2]
    # 梁は、この2点を結んで描画。角度は検出角度（15度刻み）
    adj_angle = round_angle_deg(angle_deg)
    beam_connections.append({
        "nidx_a": int(idx1),
        "nidx_b": int(idx2),
        "a": node_a.tolist(),
        "b": node_b.tolist(),
        "angle_deg": adj_angle,
        "conf": float(b["conf"])
    })

# 荷重を梁上に投影して表示
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
    adj_angle = round_angle_deg(beam["angle_deg"])
    load_connections.append({
        "type": l["type"],
        "on_beam": {"nidx_a": best_beam["nidx_a"], "nidx_b": best_beam["nidx_b"]} if best_beam else None,
        "proj_pt": best_proj.tolist() if best_proj is not None else c.tolist(),
        "angle_deg": adj_angle,
        "conf": float(l["conf"])
    })

# 清書 & ビジュアライズ
cleaned = np.ones_like(img)*255
# 梁=線分描画, 角度15度刻み
for conn in beam_connections:
    pt1, pt2 = np.array(conn["a"]), np.array(conn["b"])
    cv2.line(cleaned, tuple(map(int,pt1)), tuple(map(int,pt2)), (160,160,160), 6)
    #梁テンプレ: 棒を節点同士に貼り、テンプレート回転も15度刻み
    tpl = TEMPL.get("beam")
    if tpl is not None:
        center = (pt1+pt2)/2
        factor = np.linalg.norm(pt2-pt1)/max(tpl.shape[:2])
        tpl_scaled = scale_image(tpl, factor)
        tpl_rot    = rotate_image_keep_alpha(tpl_scaled, conn["angle_deg"])
        cleaned = overlay_rgba(cleaned, tpl_rot, center)
# 支点テンプレ貼り＋節点
for i,s in enumerate(supports):
    name = s["type"]
    tpl = TEMPL.get(name)
    center = s["node"]
    if tpl is not None:
        tpl_scaled = scale_image(tpl, 0.8)
        cleaned = overlay_rgba(cleaned, tpl_scaled, center)
    cv2.circle(cleaned, tuple(map(int,center)), 10, (0,0,255), 2)
    cv2.putText(cleaned, f"N{i}", (int(center[0])+8, int(center[1])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
# 荷重テンプレ（梁上に表示、テンプレート回転: 梁に合わせ15度刻み）
for l in load_connections:
    name = l["type"]
    tpl = TEMPL.get(name)
    center = l["proj_pt"]
    angle = l["angle_deg"]
    if tpl is not None:
        tpl_scaled = scale_image(tpl, 0.9)
        tpl_rot    = rotate_image_keep_alpha(tpl_scaled, angle)
        cleaned = overlay_rgba(cleaned, tpl_rot, center)

st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB),"清書画像（支点・梁端節点接続/荷重梁上表示/15度刻み）",use_container_width=True)

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
st.success("節点設定、梁端支点接続、15度刻み角度、荷重投影（梁上表示）で清書実行完了！")