"""
clean_and_moment.py
入力:
  - images_dir: 画像フォルダ（画像と同名のラベル .txt が labels_dir にあることを想定）
  - labels_dir: ラベルフォルダ (YOLO polygon: class x1 y1 x2 y2 ... normalized)
  - class_map: dict mapping class id -> name (例: {0:'beam',1:'pin',2:'roller',3:'fixed',4:'load',5:'UDL'})
処理:
  - ポリゴンから梁の端点、角度を推定
  - ノードスナップで接続復元
  - 荷重を beam に割り当て
  - 単純支持 or 片持ち の場合にせん断/曲げ図を描く
出力:
  - out_dir/<image>_clean.png : 元画像に重ねた清書
  - out_dir/<image>_moment.png: 曲げ図
  - out_dir/<image>_structure.json: ノード要素情報
"""

import os, glob, json, math, argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ----------- ユーティリティ関数 ----------
def denorm_points(norm_pts, w, h):
    pts = []
    for i in range(0, len(norm_pts), 2):
        x = float(norm_pts[i]) * w
        y = float(norm_pts[i+1]) * h
        pts.append((x,y))
    return np.array(pts)

def polygon_centroid(pts):
    # pts: Nx2 ndarray
    x = pts[:,0]; y = pts[:,1]
    a = 0.5*np.sum(x*np.roll(y,-1)-np.roll(x,-1)*y)
    if abs(a) < 1e-8:
        return pts.mean(axis=0)
    cx = (1/(6*a))*np.sum((x+np.roll(x,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))
    cy = (1/(6*a))*np.sum((y+np.roll(y,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))
    return np.array([cx,cy])

def pca_axis_endpoints(pts):
    # returns endpoints and angle in radians (principal axis)
    # handle degenerate
    if len(pts) < 2:
        return None
    mean = pts.mean(axis=0)
    cov = np.cov((pts-mean).T)
    w, v = np.linalg.eig(cov)
    idx = np.argmax(w)
    axis = v[:,idx]
    proj = (pts-mean).dot(axis)
    a,b = proj.min(), proj.max()
    pt1 = mean + a*axis
    pt2 = mean + b*axis
    angle = math.atan2(axis[1], axis[0])
    return (pt1, pt2, angle)

def angle_deg_from_vec(vx, vy):
    d = math.degrees(math.atan2(vy, vx))
    if d < 0: d += 360
    return d

# ----------- ラベル読み込み ----------
def parse_label_file_txt(label_path, img_w, img_h):
    """
    想定フォーマット: each line: class x1 y1 x2 y2 ... (normalized 0..1)
    returns list of dict: {cls:int, points: np.array(N,2), centroid: (x,y)}
    """
    items = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            toks=line.split()
            cls=int(float(toks[0]))
            coords = [float(x) for x in toks[1:]]
            pts = denorm_points(coords, img_w, img_h)
            cen = polygon_centroid(pts)
            items.append({'cls':cls, 'pts':pts, 'centroid':cen})
    return items

# ----------- ノード / ビームの復元 ----------
def extract_beams_and_supports(img_path, label_path, class_map):
    img = Image.open(img_path)
    w,h = img.size
    labels = parse_label_file_txt(label_path, w, h)
    beams = []
    supports = []
    loads = []
    udls = []
    for it in labels:
        name = class_map.get(it['cls'], str(it['cls']))
        if name == 'beam':
            # PCA で端点推定
            res = pca_axis_endpoints(it['pts'])
            if res is None: continue
            p1,p2,angle = res
            beams.append({'pts':it['pts'], 'p1':tuple(p1),'p2':tuple(p2),'angle':angle})
        elif name in ('pin','roller','fixed'):
            supports.append({'type':name,'centroid':tuple(it['centroid'])})
        elif name in ('load','concentratedload'):
            loads.append({'type':'point','centroid':tuple(it['centroid'])})
        elif name in ('UDL','udl','load_udl','distributed'):
            udls.append({'type':'udl','pts':it['pts']})
        else:
            # unknown -> treat as point load
            loads.append({'type':'point','centroid':tuple(it['centroid'])})
    return img, beams, supports, loads, udls

def snap_nodes(endpoints, tol=12.0):
    # endpoints: list of (x,y)
    if len(endpoints)==0:
        return [], {}
    arr = np.array(endpoints)
    tree = cKDTree(arr)
    clusters = {}
    used = set()
    node_centers = []
    mapping = {}
    idx = 0
    for i,p in enumerate(arr):
        if i in used: continue
        neighbors = tree.query_ball_point(p, r=tol)
        pts = arr[neighbors]
        center = pts.mean(axis=0)
        node_centers.append(tuple(center))
        for j in neighbors:
            mapping[j]=idx
            used.add(j)
        idx+=1
    return node_centers, mapping

# ----------- beam/loads -> assignment ----------
def project_point_on_seg(p, a, b):
    # p,a,b: (x,y) tuples
    pa = np.array(p)-np.array(a)
    ab = np.array(b)-np.array(a)
    L2 = (ab**2).sum()
    if L2==0: return 0.0, tuple(a)
    t = np.dot(pa,ab)/L2
    t_clamped = max(0.0, min(1.0, t))
    proj = np.array(a) + t_clamped*ab
    return t_clamped, tuple(proj)

# ----------- 荷重・反力・曲げ図（単純梁・片持ち） ----------
def analyze_beam_simple(L, point_loads, udl_segments, support_type):
    """
    L: length (float)
    point_loads: list of tuples (P, a) where a: distance from left (m or px)
    udl_segments: list of tuples (w, a, b) where uniform load intensity w (force per length) from a..b
    support_type: 'simply' or 'cantilever'  (others -> skip)
    Returns (x_grid, V(x), M(x)) arrays
    """
    N=1001
    xs = np.linspace(0, L, N)
    dx = xs[1]-xs[0]
    # build q(x)
    q = np.zeros_like(xs)  # load per length
    # distribute point loads as concentrated bins (approx)
    for P,a in point_loads:
        # place at nearest grid index
        idx = int(round((a / L) * (N-1)))
        if idx<0: idx=0
        if idx> N-1: idx=N-1
        # convert to equivalent density for integral: add P/dx at idx
        q[idx] += P / dx
    for (w,a,b) in udl_segments:
        # fill between a..b
        i0 = int(max(0, math.floor(a/L*(N-1))))
        i1 = int(min(N-1, math.ceil(b/L*(N-1))))
        q[i0:i1+1] += w  # w is already force per length
    # total loads (should be roughly sum P + sum w*(b-a))
    total_load = 0.0
    for P,a in point_loads: total_load += P
    for w,a,b in udl_segments: total_load += w*(b-a)
    # analysis
    if support_type == 'simply':
        # compute RB by moments about left
        moment_sum = 0.0
        for P,a in point_loads:
            moment_sum += P * (L - a)
        for w,a,b in udl_segments:
            W = w*(b-a)
            x_c = a + (b-a)/2.0
            moment_sum += W * (L - x_c)
        RB = moment_sum / L
        RA = total_load - RB
        # shear from left
        V = np.zeros_like(xs)
        cum = 0.0
        # prepare arrays of point loads cumulative
        ploads_by_idx = np.zeros_like(xs)
        for P,a in point_loads:
            idx = int(round((a / L) * (N-1)))
            if idx<0: idx=0
            if idx> N-1: idx=N-1
            ploads_by_idx[idx] += P
        udl_by_idx = q.copy()
        cumulative = 0.0
        V[0] = RA
        for i in range(1,N):
            # integrate q from xs[i-1] to xs[i] and subtract point loads approximated at index
            cumulative += ploads_by_idx[i-1] + udl_by_idx[i-1]*dx
            V[i] = RA - cumulative
        # moment integrate V
        M = np.cumsum(V[:-1])*dx
        M = np.concatenate([M, M[-1:]])  # same length
        # shift to have M(0)=0
        M = np.array(M)
        return xs, V, M
    elif support_type == 'cantilever':
        # for cantilever fixed at left: use integration from right:
        # define q density as above
        # compute shear V(x) = - integral from x to L of q(s) ds -> do cumulative from right
        cumulative_from_right = np.cumsum(q[::-1]) * dx
        V = -cumulative_from_right[::-1]
        # moment M(x) = - integral from x to L of V(s) ds
        M = -np.cumsum(V[::-1])*dx
        M = M[::-1]
        return xs, V, M
    else:
        return xs, np.zeros_like(xs), np.zeros_like(xs)

# ----------- 描画 ----------
def draw_structure_and_moment(img, beams, nodes, supports_by_node, loads_on_beams, udls_on_beams, results, out_prefix):
    # img: PIL.Image
    draw = ImageDraw.Draw(img)
    # draw beams
    for b in beams:
        draw.line([b['p1'], b['p2']], fill=(255,0,0), width=3)
        # angle label
        vx = b['p2'][0]-b['p1'][0]; vy = b['p2'][1]-b['p1'][1]
        ang = angle_deg_from_vec(vx,vy)
        mx = (b['p1'][0]+b['p2'][0])/2
        my = (b['p1'][1]+b['p2'][1])/2
        draw.text((mx+4,my+4), f"{ang:.0f}°", fill=(255,0,0))
    # nodes
    for i,n in enumerate(nodes):
        x,y = n
        r=4
        draw.ellipse([x-r,y-r,x+r,y+r], fill=(0,0,255))
        draw.text((x+6,y+6), str(i), fill=(0,0,255))
    # supports
    for nid, sup in supports_by_node.items():
        x,y = nodes[nid]
        tx = x; ty = y+12
        if sup == 'pin':
            # triangle
            draw.polygon([(x-8,y+12),(x+8,y+12),(x,y+2)], fill=(0,128,0))
        elif sup == 'roller':
            draw.polygon([(x-12,y+12),(x+12,y+12),(x,y+2)], fill=(128,0,128))
            # small gap: draw circle
            draw.ellipse([x-6,y+12+2,x+6,y+12+8], outline=(128,0,128))
        elif sup == 'fixed':
            # short hatch lines
            draw.line([(x-10,y+2),(x+10,y+2)], fill=(0,0,0), width=4)
            for i in range(-8,9,4):
                draw.line([(x+i,y+6),(x+i,y+12)], fill=(0,0,0))
    # save annotated image
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    img.save(out_prefix + "_clean.png")

    # draw moment diagram under it (simple)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,3))
    # average across beams results: plot each
    for beam_id,res in results.items():
        xs, V, M = res['xs'], res['V'], res['M']
        L = xs[-1]
        ax.plot(np.linspace(0,L,len(M)), M, label=f"beam{beam_id}")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("M (arb)")
    ax.axhline(0,color='k',linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_prefix + "_moment.png", dpi=150)
    plt.close(fig)

# ----------- メイン処理 ----------
def process_image(img_path, label_path, class_map, out_dir, snap_tol=12):
    img, beams, supports, loads, udls = extract_beams_and_supports(img_path, label_path, class_map)
    # endpoints list
    endpoints = []
    for b in beams:
        endpoints.append(b['p1'])
        endpoints.append(b['p2'])
    nodes_centers, mapping = snap_nodes(endpoints, tol=snap_tol)
    # mapping maps endpoint index -> node id; build beam->node mapping
    beam_nodes = []
    for i,b in enumerate(beams):
        idx1 = 2*i
        idx2 = 2*i+1
        n1 = mapping.get(idx1)
        n2 = mapping.get(idx2)
        if n1 is None or n2 is None:
            continue
        beam_nodes.append((n1,n2))
    # supports -> assign to nearest node
    nodes_arr = np.array(nodes_centers)
    if len(nodes_arr)>0:
        tree = cKDTree(nodes_arr)
    supports_by_node = {}
    for s in supports:
        if len(nodes_arr)==0: continue
        d, idx = tree.query(np.array(s['centroid']))
        if d < snap_tol*2:
            # map type
            supports_by_node[idx] = s['type']
    # loads: project onto nearest beam
    loads_assigned = []
    udls_assigned = []
    for ld in loads:
        # find beam whose segment projects near centroid
        best = None; best_dist=1e9; best_t=0.0; best_bid=None
        for i,b in enumerate(beams):
            a=b['p1'];bb=b['p2']
            t, proj = project_point_on_seg(ld['centroid'], a, bb)
            # distance:
            d = math.hypot(proj[0]-ld['centroid'][0], proj[1]-ld['centroid'][1])
            if d < best_dist:
                best_dist=d; best=(proj); best_t=t; best_bid=i
        if best_dist < 30: # px threshold
            # convert t to distance along beam
            a=np.array(beams[best_bid]['p1']); b=np.array(beams[best_bid]['p2'])
            L = np.linalg.norm(b-a)
            a_dist = best_t * L
            loads_assigned.append({'beam':best_bid, 'P': -100.0, 'a':a_dist})  # P sign arbitrary scale: user should set magnitudes
    # UDLs from udl polygons: convert to projected intervals
    for ud in udls:
        # use polygon centroid & PCA to detect beam candidate (naive)
        pts = ud['pts']
        cen = polygon_centroid(pts)
        best = None; best_dist=1e9; best_bid=None; best_proj_range=None
        for i,b in enumerate(beams):
            a=np.array(b['p1']); bb=np.array(b['p2'])
            # project all ud pts onto beam axis param t
            ab = bb - a; L = np.linalg.norm(ab)
            if L==0: continue
            unit = ab / L
            ts = []
            for p in pts:
                pa = np.array(p) - a
                t = np.dot(pa, unit) / L
                ts.append(t)
            tmin = max(0.0, min(ts)); tmax = min(1.0, max(ts))
            # compute polygon centroid dist to projected center
            proj_center = a + ((tmin+tmax)/2.0)*ab
            d = np.linalg.norm(proj_center - cen)
            if d < best_dist:
                best_dist=d; best_bid=i; best_proj_range=(tmin,tmax)
        if best_bid is not None:
            a=np.array(beams[best_bid]['p1']); bb=np.array(beams[best_bid]['p2'])
            L = np.linalg.norm(bb-a)
            a_px = best_proj_range[0]*L
            b_px = best_proj_range[1]*L
            udls_assigned.append({'beam':best_bid, 'w': -10.0, 'a':a_px, 'b':b_px})
    # now analyze each beam
    results = {}
    for i,b in enumerate(beams):
        n1, n2 = beam_nodes[i] if i < len(beam_nodes) else (None,None)
        a = np.array(b['p1']); bb = np.array(b['p2'])
        L = np.linalg.norm(bb - a)
        # check supports at endpoints
        s1 = supports_by_node.get(n1)
        s2 = supports_by_node.get(n2)
        support_type = 'free'
        if (s1 in ('pin','roller') and s2 in ('pin','roller')):
            support_type = 'simply'
        elif (s1 == 'fixed' and (s2 is None)):
            support_type = 'cantilever'
        elif (s2 == 'fixed' and (s1 is None)):
            # reverse beam orientation to make left as fixed
            # swap endpoints for calculation (we'll keep visuals intact)
            # implement by flipping arrays and later restore
            support_type = 'cantilever'
        else:
            # if one end supported, one not -> treat as cantilever approx
            if (s1 in ('pin','roller') and s2 is None) or (s2 in ('pin','roller') and s1 is None):
                support_type = 'cantilever'  # approximate
            else:
                support_type = 'unknown'
        # collect loads for this beam
        pls = []
        uds = []
        for ld in loads_assigned:
            if ld['beam'] == i:
                pls.append((ld['P'], ld['a']))
        for ud in udls_assigned:
            if ud['beam'] == i:
                uds.append((ud['w'], ud['a'], ud['b']))
        if support_type in ('simply','cantilever'):
            xs,V,M = analyze_beam_simple(L, pls, uds, support_type)
            results[i] = {'xs':xs, 'V':V, 'M':M}
        else:
            results[i] = {'xs':np.array([0,L]), 'V':np.zeros(2), 'M':np.zeros(2)}
    # draw and save
    base = os.path.basename(img_path)
    name = os.path.splitext(base)[0]
    out_prefix = os.path.join(out_dir, name)
    draw_structure_and_moment(img.copy(), beams, nodes_centers, supports_by_node, loads_assigned, udls_assigned, results, out_prefix)
    # write json summary
    summary = {
        'nodes': nodes_centers,
        'beams': [ {'p1':b['p1'],'p2':b['p2'],'angle_deg': angle_deg_from_vec(b['p2'][0]-b['p1'][0], b['p2'][1]-b['p1'][1])} for b in beams],
        'supports': supports_by_node,
        'loads': loads_assigned,
        'udls': udls_assigned
    }
    with open(out_prefix + "_structure.json","w",encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary, results

# ----------- コマンドライン ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True)
    parser.add_argument('--labels_dir', required=True)
    parser.add_argument('--out_dir', default='out')
    parser.add_argument('--snap_tol', type=float, default=12.0)
    args = parser.parse_args()
    images = sorted(glob.glob(os.path.join(args.images_dir, "*.*")))
    os.makedirs(args.out_dir, exist_ok=True)
    # class map example: set accordingly to your dataset mapping
    class_map = {0:'beam', 1:'pin', 2:'roller', 3:'fixed', 4:'load', 5:'UDL'}
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(args.labels_dir, name + ".txt")
        if not os.path.exists(label_path):
            print("label missing for", img_path); continue
        print("processing", name)
        summary, results = process_image(img_path, label_path, class_map, args.out_dir, snap_tol=args.snap_tol)
        print("done:", name)
if __name__=='__main__':
    main()
