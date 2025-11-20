# labeling_rules_v2.py
import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(layout="wide")

# ---------------------------
# ヘルパー関数群
# ---------------------------
def load_template(path):
    t = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if t is None:
        st.warning(f"テンプレートが読み込めません: {path}")
        return None
    if t.shape[2] == 3:
        b,g,r = cv2.split(t)
        alpha = np.ones_like(b) * 255
        t = cv2.merge([b,g,r,alpha])
    return t

def overlay_template(base_img, template_img, center, angle_deg, scale=1.0):
    """RGBA template を base_img に回転・スケールして貼る（角度は度、CCW正）"""
    if template_img is None:
        return base_img
    th, tw = template_img.shape[:2]
    new_w, new_h = max(int(tw * scale), 2), max(int(th * scale), 2)
    tpl = cv2.resize(template_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    M = cv2.getRotationMatrix2D((new_w//2, new_h//2), angle_deg, 1.0)
    rotated = cv2.warpAffine(tpl, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    x, y = int(center[0]), int(center[1])
    x1, y1 = max(x - new_w//2, 0), max(y - new_h//2, 0)
    x2, y2 = min(x1 + new_w, base_img.shape[1]), min(y1 + new_h, base_img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return base_img
    roi = base_img[y1:y2, x1:x2].astype(float)
    rot_crop = rotated[0:(y2-y1), 0:(x2-x1)]
    if rot_crop.shape[2] < 4:
        base_img[y1:y2, x1:x2] = rot_crop[..., :3]
        return base_img
    alpha = rot_crop[..., 3:4] / 255.0
    base_img[y1:y2, x1:x2] = (1-alpha)*roi + alpha*rot_crop[..., :3]
    base_img[y1:y2, x1:x2] = base_img[y1:y2, x1:x2].astype(np.uint8)
    return base_img

def polygon_center(pts):
    return pts.mean(axis=0)

def edge_list_from_pts(pts):
    # pts shape (4,2) in order (not necessarily ordered) — edges: (0-1),(1-2),(2-3),(3-0)
    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        angle = math.degrees(math.atan2(vec[1], vec[0]))  # CCW degrees
        midpoint = (p1 + p2) / 2.0
        edges.append({"i": i, "p1": p1, "p2": p2, "vec": vec, "len": length, "angle": angle, "mid": midpoint})
    return edges

def short_edge_indices(edges):
    lens = [e["len"] for e in edges]
    sorted_idx = np.argsort(lens)  # ascending
    return sorted_idx[:2].tolist()  # two shortest edges indices

def long_edge_indices(edges):
    lens = [e["len"] for e in edges]
    sorted_idx = np.argsort(lens)
    return sorted_idx[-2:].tolist()

def midpoint_of_edges(edges, idx):
    e = edges[idx]
    return e["mid"]

def distance_point_to_segment(p, a, b):
    # Euclidean distance from point p to segment a-b
    pa = p - a
    ba = b - a
    t = np.dot(pa, ba) / (np.dot(ba, ba) + 1e-12)
    t = max(0.0, min(1.0, t))
    proj = a + t*ba
    return np.linalg.norm(p - proj)

def line_from_two_points(a, b):
    # returns (point, direction unit vector)
    v = b - a
    norm = np.linalg.norm(v) + 1e-12
    return a, v / norm

def project_point_to_line(p, a, dir_unit):
    # returns scalar projection distance along dir_unit from a to proj
    return np.dot(p - a, dir_unit)

# ---------------------------
# 頂点順序正規化（角度ソート→clockwise）
# ---------------------------
def order_points_clockwise(pts):
    c = polygon_center(pts)
    angs = np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0])
    order = np.argsort(angs)  # CCW ascending
    pts_ccw = pts[order]
    pts_cw = pts_ccw[::-1]
    return pts_cw

# ---------------------------
# 再配置ルール
# ---------------------------
def compute_placement_for_element(pts_raw, cls_name, beam_lines):
    """
    pts_raw: (4,2) polygon from OBB
    cls_name: normalized class name (lower, no spaces)
    beam_lines: list of dicts for detected beams { 'a','b','dir','centerline_a','dir_unit' }
    returns dict with keys: center (x,y), angle (deg CCW), scale, extra (like tip_idx)
    """
    pts = order_points_clockwise(np.array(pts_raw, dtype=float))
    edges = edge_list_from_pts(pts)
    short_idxs = short_edge_indices(edges)
    long_idxs = long_edge_indices(edges)

    # Helper to find beam nearest to this element (by distance from element center to beam centerline)
    center = polygon_center(pts)
    nearest_beam = None
    min_dist = 1e9
    for b in beam_lines:
        # distance from center to beam line segment
        d = distance_point_to_segment(center, b["a"], b["b"])
        if d < min_dist:
            min_dist = d
            nearest_beam = b

    result = {"class": cls_name, "center": center, "angle": 0.0, "scale": 1.0, "pts": pts}

    # ---------- BEAM ----------
    if "beam" in cls_name:
        # beam centerline = connect midpoints of the two short edges
        m1 = midpoint_of_edges(edges, short_idxs[0])
        m2 = midpoint_of_edges(edges, short_idxs[1])
        centerline_a, dir_unit = line_from_two_points(m1, m2)
        angle_deg = math.degrees(math.atan2(dir_unit[1], dir_unit[0]))  # CCW degrees
        # scale: match template long side to long edge length (choose larger edge length)
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        result.update({"center": (m1 + m2)/2.0, "angle": angle_deg, "scale": long_len, "centerline": (m1,m2)})
        return result

    # ---------- PIN / ROLLER ----------
    if cls_name in ["pin", "roller"]:
        # find the box edge most parallel to nearest beam (if present). That edge side will be "down".
        if nearest_beam is not None:
            beam_dir = nearest_beam["dir_unit"]
            # compute absolute angle difference between each edge direction and beam_dir
            best = None
            best_diff = 1e9
            for e in edges:
                ev = e["vec"] / (np.linalg.norm(e["vec"]) + 1e-12)
                diff = abs(np.dot(ev, beam_dir))  # 1 means parallel
                if diff > best_diff if False else True:  # we'll compute properly below
                    pass
            # re-evaluate properly: choose edge with maximum |dot| with beam_dir
            best = None
            best_dot = -1.0
            for e in edges:
                ev = e["vec"] / (np.linalg.norm(e["vec"]) + 1e-12)
                dot = abs(np.dot(ev, beam_dir))
                if dot > best_dot:
                    best_dot = dot
                    best = e
            # base edge is best; position template so that its 'down' aligns to outward normal of that edge (toward outside)
            base = best
            # compute edge normal pointing inward: normal = rotate ev by +90
            ev = base["vec"] / (np.linalg.norm(base["vec"]) + 1e-12)
            normal = np.array([-ev[1], ev[0]])  # one normal
            # determine which normal points toward polygon center: check sign
            # edge midpoint -> normal outward? We want 'down' side = side with that edge present.
            # We'll compute angle such that template's "down" points toward base edge (i.e., normal points from center to edge)
            # vector from center to edge midpoint:
            v_center_to_mid = base["mid"] - center
            # choose normal direction that points from center to edge (i.e., same sign)
            if np.dot(normal, v_center_to_mid) < 0:
                normal = -normal
            # angle: normal points outward; template 'down' direction equals -90 deg from template's "right"?
            angle_deg = math.degrees(math.atan2(-normal[1], -normal[0]))  # point template so tip up/down accordingly
            # approximate scale by average short edge length
            avg_short = (edges[short_idxs[0]]["len"] + edges[short_idxs[1]]["len"]) / 2.0
            result.update({"center": center + normal* (avg_short*0.05), "angle": angle_deg, "scale": avg_short/40.0})
            return result
        else:
            # no beam: fallback: take topmost point as tip -> place template with tip pointing to top
            top_idx = int(np.argmin(pts[:,1]))
            tip = pts[top_idx]
            centerline_dir = pts[(top_idx+2)%4] - tip
            angle_deg = math.degrees(math.atan2(centerline_dir[1], centerline_dir[0]))
            avg_short = (edges[short_idxs[0]]["len"] + edges[short_idxs[1]]["len"]) / 2.0
            result.update({"center": center, "angle": angle_deg, "scale": avg_short/40.0})
            return result

    # ---------- LOAD ----------
    if cls_name in ["load", "kajyu"]:
        # short-edges midpoints -> central segment = arrow shaft
        m1 = midpoint_of_edges(edges, short_idxs[0])
        m2 = midpoint_of_edges(edges, short_idxs[1])
        seg_mid = (m1 + m2)/2.0
        shaft_dir = (m2 - m1)
        if np.linalg.norm(shaft_dir) < 1e-6:
            shaft_dir = np.array([1.0, 0.0])
        dir_unit = shaft_dir / (np.linalg.norm(shaft_dir) + 1e-12)
        # decide which end (m1 or m2) is closer to a beam -> that end should have the arrowhead (矢じり)
        if nearest_beam is not None:
            d1 = distance_point_to_segment(m1, nearest_beam["a"], nearest_beam["b"])
            d2 = distance_point_to_segment(m2, nearest_beam["a"], nearest_beam["b"])
            if d1 < d2:
                tip_point = m1
                tail_point = m2
                dir_unit_final = (tip_point - tail_point)
            else:
                tip_point = m2
                tail_point = m1
                dir_unit_final = (tip_point - tail_point)
        else:
            # fallback: use shorter distance from segment midpoint to polygon center
            tip_point = m1 if np.linalg.norm(m1-center) < np.linalg.norm(m2-center) else m2
            tail_point = m2 if np.all(tip_point==m1) else m1
            dir_unit_final = (tip_point - tail_point)
        if np.linalg.norm(dir_unit_final) < 1e-6:
            dir_unit_final = shaft_dir
        angle_deg = math.degrees(math.atan2(dir_unit_final[1], dir_unit_final[0]))
        # scale based on long edge length
        long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
        result.update({"center": seg_mid, "angle": angle_deg, "scale": long_len/40.0, "tip": tip_point})
        return result

    # ---------- UDL ----------
    if cls_name in ["udl"]:
        # we want: template's left side to be on the edge which is far from beam and mostly parallel to beam
        # find edge almost parallel to nearest_beam dir and choose the one far from beam
        if nearest_beam is not None:
            beam_dir = nearest_beam["dir_unit"]
            best_edge = None
            best_dot = -1.0
            for e in edges:
                ev = e["vec"] / (np.linalg.norm(e["vec"])+1e-12)
                dot = abs(np.dot(ev, beam_dir))
                if dot > best_dot:
                    best_dot = dot
                    best_edge = e
            # determine side nearer/farther to beam: midpoint distance
            d = distance_point_to_segment(best_edge["mid"], nearest_beam["a"], nearest_beam["b"])
            # choose orientation so that template left side is on the edge far from beam -> if mid is far, left=that side
            angle_deg = best_edge["angle"]
            # center place: put template center on polygon center
            long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
            result.update({"center": center, "angle": angle_deg, "scale": long_len/40.0})
            return result
        else:
            # fallback simple
            long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
            angle_deg = edges[long_idxs[0]]["angle"]
            result.update({"center": center, "angle": angle_deg, "scale": long_len/40.0})
            return result

    # ---------- FIXED ----------
    if cls_name in ["fixed"]:
        # choose the edge mostly perpendicular to beam and the one closer to beam is template's right side
        if nearest_beam is not None:
            beam_dir = nearest_beam["dir_unit"]
            # find edge with minimal absolute dot (perpendicular)
            best_edge = None
            best_val = 1e9
            for e in edges:
                ev = e["vec"] / (np.linalg.norm(e["vec"])+1e-12)
                val = abs(abs(np.dot(ev, beam_dir)) - 0.0)  # small if perpendicular
                if val < best_val:
                    best_val = val
                    best_edge = e
            # determine which side of that edge is nearer to beam (the nearer side becomes template's 'right side')
            # We'll compute angle so that template's right side aligns with that edge pointing toward beam
            mid = best_edge["mid"]
            # vector from mid to beam center
            if nearest_beam is not None:
                beam_center = (nearest_beam["a"] + nearest_beam["b"]) / 2.0
                v_mid_to_beam = beam_center - mid
                # angle for template: set template such that its right side points toward beam,
                # so template rotation = angle of edge + 90 if needed:
                edge_angle = best_edge["angle"]
                # check sign: if projecting edge normal toward beam positive then rotate accordingly
                angle_deg = edge_angle  # baseline
                if np.dot(v_mid_to_beam, np.array([-best_edge["vec"][1], best_edge["vec"][0]])) < 0:
                    angle_deg += 180
            else:
                angle_deg = best_edge["angle"]
            long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
            result.update({"center": center, "angle": angle_deg, "scale": long_len/40.0})
            return result
        else:
            # fallback
            long_len = max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])
            angle_deg = edges[long_idxs[0]]["angle"]
            result.update({"center": center, "angle": angle_deg, "scale": long_len/40.0})
            return result

    # ---------- other default ----------
    # fallback: use short-edge midpoints center and edge 0-1 angle
    m1 = midpoint_of_edges(edges, short_idxs[0])
    m2 = midpoint_of_edges(edges, short_idxs[1])
    angle_deg = math.degrees(math.atan2((m2-m1)[1], (m2-m1)[0]))
    result.update({"center": (m1+m2)/2.0, "angle": angle_deg, "scale": max(edges[long_idxs[0]]["len"], edges[long_idxs[1]]["len"])/40.0})
    return result

# ---------------------------
# テンプレート群ロード
# ---------------------------
TEMPLATES = {
    "pin": load_template("templates/pin.png"),
    "roller": load_template("templates/roller.png"),
    "hinge": load_template("templates/hinge.png"),
    "fixed": load_template("templates/fixed.png"),
    "beam": load_template("templates/beam.png"),
    "load": load_template("templates/load.png"),
    "momentl": load_template("templates/momentL.png"),
    "momentr": load_template("templates/momentR.png"),
    "udl": load_template("templates/UDL.png"),
}

# ---------------------------
# main
# ---------------------------
def main():
    st.title("🧩 構造図 再配置ルールV2（beam/支点/荷重/UDL/fixed）")
    model_path = st.text_input("YOLO OBB model path", value=r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt")
    conf_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.01)

    uploaded = st.file_uploader("構造図をアップロード", type=["png","jpg","jpeg"])
    if uploaded is None:
        st.info("画像をアップロードしてください")
        return

    pil = Image.open(uploaded).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    st.image(pil, caption="入力画像", use_container_width=True)

    if not os.path.exists(model_path):
        st.warning("モデルファイルが見つかりません。モデルパスを確認してください。")
        return

    # load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"モデルロードエラー: {e}")
        return

    if st.button("実行: 検出→順序補正→再配置"):
        res = model(img, conf=conf_th, imgsz=640)[0]

        vis = img.copy()
        cleaned = np.ones_like(img) * 255

        # first pass: collect beams to build beam_lines (for near relations)
        beam_lines = []
        elements = []
        if hasattr(res, "obb") and res.obb is not None:
            obb = res.obb
            for i in range(len(obb)):
                # skip if None
                try:
                    cls_id = int(obb.cls[i].cpu().numpy())
                    conf = float(obb.conf[i].cpu().numpy())
                except Exception:
                    continue
                if conf < conf_th:
                    continue
                name = res.names[cls_id].lower().replace(" ", "")
                xy8 = obb.xyxyxyxy[i].cpu().numpy().reshape(4,2)
                pts = order_points_clockwise(xy8)
                edges = edge_list_from_pts(pts)
                # compute short edges midpoints and beam centerline for beams
                if "beam" in name:
                    sidx = short_edge_indices(edges)
                    m1 = midpoint_of_edges(edges, sidx[0])
                    m2 = midpoint_of_edges(edges, sidx[1])
                    a = m1; b = m2
                    dir_unit = (b-a)/ (np.linalg.norm(b-a)+1e-12)
                    beam_lines.append({"a":a, "b":b, "dir_unit":dir_unit, "pts":pts})
                elements.append({"name":name,"pts":pts,"conf":conf,"raw_idx":i})

        # second pass: compute placements using beam_lines
        placements = []
        for elem in elements:
            placement = compute_placement_for_element(elem["pts"], elem["name"], beam_lines)
            placements.append({"elem":elem, "placement":placement})

            # draw detection polygon + vertex indices on vis for debugging
            pts = elem["pts"].astype(np.int32)
            cv2.polylines(vis, [pts], True, (0,200,0), 2)
            for vi, p in enumerate(pts):
                cv2.circle(vis, tuple(p.tolist()), 4, (0,0,255), -1)
                cv2.putText(vis, str(vi+1), (int(p[0]+6), int(p[1]+6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # draw beam centerlines too
        for b in beam_lines:
            a = tuple(b["a"].astype(int)); c = tuple(b["b"].astype(int))
            cv2.line(vis, a, c, (255,0,0), 2)
            cv2.circle(vis, a, 3, (255,0,0), -1); cv2.circle(vis, c, 3, (255,0,0), -1)

        # show detection visualization
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="検出可視化（ポリゴン＋頂点＋ビーム線）", use_container_width=True)

        # apply templates according to placements
        for p in placements:
            cls = p["elem"]["name"]
            place = p["placement"]
            tpl = TEMPLATES.get(cls)
            if tpl is None:
                # skip unknown
                continue
            center = tuple(place["center"])
            angle = float(place["angle"])
            # scale: note we used raw lengths as scale measure; normalize by an empirical factor
            scale = float(place.get("scale", 1.0))
            # normalize: if scale looks like pixel length, convert to template scale factor:
            # assume template long side ~ template.shape[1] (width)
            tpl_h, tpl_w = tpl.shape[:2]
            # if scale was computed as pixel-length, convert to factor:
            if scale > 10:  # heuristics: large means pixel length -> convert
                factor = max(scale / max(tpl_w, tpl_h), 0.5)
            else:
                factor = max(scale, 0.4)
            cleaned = overlay_template(cleaned, tpl, center, angle, factor)

            # draw debug marks: center and angle arrow
            cx, cy = int(center[0]), int(center[1])
            cv2.circle(cleaned, (cx,cy), 4, (0,0,255), -1)
            # small direction line
            dx = int(20 * math.cos(math.radians(angle)))
            dy = int(20 * math.sin(math.radians(angle)))
            cv2.line(cleaned, (cx,cy), (cx+dx, cy+dy), (0,0,255), 2)

        st.image(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB), caption="清書結果（テンプレート貼付）", use_container_width=True)

        # log placements
        st.subheader("配置一覧（デバッグ）")
        for p in placements:
            cls = p["elem"]["name"]
            place = p["placement"]
            st.write(f"{cls}: center=({place['center'][0]:.1f},{place['center'][1]:.1f}) angle={place['angle']:.1f} scale_raw={place.get('scale',0):.1f}")

if __name__ == "__main__":
    main()
