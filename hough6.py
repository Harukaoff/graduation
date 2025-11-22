import cv2
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.cluster import DBSCAN

def detect_triangles(image_np, blur_kernel, threshold_type, block_size, C_value, canny_lower, canny_upper, min_area, approx_epsilon_factor):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    # 二値化
    if threshold_type == "Otsu":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_type == "Adaptive Mean":
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, C_value)
    elif threshold_type == "Adaptive Gaussian":
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, C_value)
    else:
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(binary, canny_lower, canny_upper)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        epsilon = approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            pts = approx.reshape(-1,2)
            triangles.append(pts)
    return triangles, edges

def detect_lines(edges, hough_threshold, min_beam_len, max_gap):
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold, minLineLength=min_beam_len, maxLineGap=max_gap)
    line_segs = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            line_segs.append(((x1, y1), (x2, y2)))
    return line_segs

def find_intersections(lines, img_shape):
    points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = line_intersection(lines[i], lines[j])
            if pt is not None and 0 <= pt[0] < img_shape[1] and 0 <= pt[1] < img_shape[0]:
                points.append(pt)
    return points

def line_intersection(l1, l2):
    (x1, y1), (x2, y2) = l1
    (x3, y3), (x4, y4) = l2
    denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if denom == 0: return None
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
    return int(px), int(py)

def cluster_points(points, eps=20, min_samples=3):
    if not points:
        return []
    points_np = np.array(points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np)
    labels = clustering.labels_
    clusters = []
    for lbl in set(labels):
        if lbl == -1: continue
        cluster = points_np[labels == lbl]
        center = tuple(np.mean(cluster, axis=0).astype(int))
        clusters.append(center)
    return clusters

def get_endpoint_centers(triangles):
    triangles = sorted(triangles, key=lambda pts: pts[np.argmin(pts[:,0])][0])
    left_tri = triangles[0]
    right_tri = triangles[-1]
    def bottom_center(pts):
        idxs = np.argsort(pts[:,1])[-2:]
        bottom_pts = pts[idxs]
        center = np.mean(bottom_pts, axis=0)
        return tuple(center.astype(int))
    return bottom_center(left_tri), bottom_center(right_tri)

def align_triangles_to_beam(image_np, triangles, y_beam):
    img_out = image_np.copy()
    new_triangles = []
    for pts in triangles:
        top_idx = np.argmin(pts[:,1])
        apex = pts[top_idx]
        dy = y_beam - apex[1]
        pts_moved = pts + np.array([0, dy])
        pts_moved = pts_moved.astype(int)
        new_triangles.append(pts_moved)
        cv2.drawContours(img_out, [pts_moved], 0, (0,255,0), 3)
        cv2.circle(img_out, tuple(pts_moved[top_idx]), 8, (255,0,0), -1)
    return img_out, new_triangles

def classify_support(triangle_pts, beam_y, fixed_clusters, support_thresh=15, fixed_thresh=30):
    support_types = []
    for pts in triangle_pts:
        # 下端点
        bottom_idx = np.argmax(pts[:,1])
        bottom_pt = pts[bottom_idx]
        # 固定支点判定: fixed_clusters（交点クラスタ）に近ければ固定
        is_fixed = False
        for fc in fixed_clusters:
            if np.linalg.norm(np.array(fc) - np.array(bottom_pt)) < fixed_thresh:
                is_fixed = True
                break
        # 梁に近ければピンローラー
        if is_fixed:
            support_types.append("固定")
        elif abs(bottom_pt[1] - beam_y) < support_thresh:
            support_types.append("ピンローラー")
        else:
            support_types.append("ピン")
    return support_types

st.set_page_config(layout="wide", page_title="三角形・梁・支点自動認識アプリ")
st.title("三角形・水平梁・支点（ピン/ピンローラー/固定）自動認識アプリ")
st.write("画像から三角形を検出し、水平梁に整列。直線検出＆交点クラスタリングで固定支点も自動認識。端部支点中央から梁を出します。")

uploaded_file = st.file_uploader("画像をアップロード", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    st.sidebar.header("パラメータ調整")
    blur_kernel = st.sidebar.slider("ガウシアンブラー カーネルサイズ", 1, 21, 5, 2)
    threshold_method = st.sidebar.selectbox("二値化方法", ("Adaptive Gaussian", "Adaptive Mean", "Otsu", "Simple Binary"))
    block_size = st.sidebar.slider("適応的閾値のブロックサイズ", 3, 51, 29, 2)
    C_value = st.sidebar.slider("適応的閾値のC値", -10, 10, 2, 1)
    canny_lower = st.sidebar.slider("Canny下限閾値", 0, 255, 50, 1)
    canny_upper = st.sidebar.slider("Canny上限閾値", 0, 255, 150, 1)
    min_area = st.sidebar.slider("最小三角形面積 (px^2)", 10, 5000, 1720, 10)
    approx_epsilon = st.sidebar.slider("輪郭近似の精度 (0.01 - 0.20)", 0.01, 0.20, 0.2, 0.01, format="%.2f")
    hough_threshold = st.sidebar.slider("Hough検出閾値", 10, 200, 25, 1)
    min_beam_len = st.sidebar.slider("最小ビーム長", 10, 1000, 500, 10)
    max_gap = st.sidebar.slider("最大ギャップ長", 1, 50, 16, 1)
    beam_ratio = st.sidebar.slider("ビーム長の割合(最大長に対して)", 0.3, 1.0, 0.7, 0.01)
    support_thresh = st.sidebar.slider("支点判定のy座標閾値", 1, 50, 15, 1)
    fixed_thresh = st.sidebar.slider("固定支点判定の距離閾値", 10, 100, 30, 1)
    cluster_eps = st.sidebar.slider("交点クラスタリング半径", 5, 100, 20, 1)
    cluster_min = st.sidebar.slider("固定支点クラスタ最小交点数", 2, 10, 3, 1)

    img = Image.open(uploaded_file)
    image_np = np.array(img.convert('RGB'))
    triangles, edges = detect_triangles(
        image_np, blur_kernel, threshold_method, block_size,
        C_value, canny_lower, canny_upper, min_area, approx_epsilon)
    # 梁y座標決定（整列用）: 三角形上端の平均y
    apexes = [tuple(pts[np.argmin(pts[:,1])]) for pts in triangles]
    if len(apexes) >= 2:
        y_beam = int(np.mean([apex[1] for apex in apexes]))
        x1, x2 = min([apex[0] for apex in apexes]), max([apex[0] for apex in apexes])
        # 端部中央（底辺中央）
        left_center, right_center = get_endpoint_centers(triangles)
        pt1, pt2 = left_center, right_center
        # 梁は端部底辺中央のy座標で完全水平
        beam_y = int(np.mean([pt1[1], pt2[1]]))
        pt1 = (pt1[0], beam_y)
        pt2 = (pt2[0], beam_y)
        # 三角形を梁に整列
        img_tri, triangles_aligned = align_triangles_to_beam(image_np, triangles, beam_y)
        # 梁描画
        cv2.line(img_tri, pt1, pt2, (0,0,255), 6)
        cv2.circle(img_tri, pt1, 12, (0,255,255), -1)
        cv2.circle(img_tri, pt2, 12, (0,255,255), -1)
        # 直線検出＆固定支点判定
        lines = detect_lines(edges, hough_threshold, min_beam_len, max_gap)
        intersections = find_intersections(lines, img_tri.shape)
        fixed_clusters = cluster_points(intersections, eps=cluster_eps, min_samples=cluster_min)
        # 固定支点描画
        for fc in fixed_clusters:
            cv2.circle(img_tri, fc, 14, (0,0,128), -1)
            cv2.putText(img_tri, "固定", fc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # 支点種別判定
        supports = classify_support(triangles_aligned, beam_y, fixed_clusters, support_thresh, fixed_thresh)
        # ラベル描画
        for i, pts in enumerate(triangles_aligned):
            bottom_idx = np.argmax(pts[:,1])
            bottom_pt = tuple(pts[bottom_idx])
            label = supports[i]
            color = (0,0,255) if label=="ピンローラー" else (0,128,0) if label=="ピン" else (0,0,128)
            cv2.putText(img_tri, label, bottom_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        st.image(img_tri, caption="三角形・水平梁・支点（ピン/ピンローラー/固定）判定", use_column_width=True)
        st.write("三角形数:", len(triangles))
        st.write("支点判定:", supports)
        st.write("固定支点（交点クラスタ）座標:", fixed_clusters)
    else:
        st.write("三角形が2つ以上検出されませんでした。")