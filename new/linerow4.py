import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.title("構造図の清書・可視化（白黒二極化過程＋三角形検出＋水平梁一本化）")

uploaded_file = st.file_uploader(
    "構造図画像をアップロードしてください",
    type=["png", "jpg", "jpeg"],
    key="main_uploader"
)

if uploaded_file is not None:
    # --- 画像読込＆元画像表示 ---
    pil_img = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_img)
    st.subheader("元画像")
    st.image(image, channels="RGB")

    # --- グレースケール変換 ---
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    st.subheader("グレースケール画像")
    st.image(gray, clamp=True, channels="GRAY")

    # --- 最黒・最白画素値検出および位置可視化 ---
    min_val = int(np.min(gray))
    max_val = int(np.max(gray))
    st.write(f"最も黒い画素値: {min_val}")
    st.write(f"最も白い画素値: {max_val}")

    vis_points = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    min_coords = np.column_stack(np.where(gray == min_val))
    max_coords = np.column_stack(np.where(gray == max_val))
    if len(min_coords) > 0:
        vis_points[min_coords[:,0], min_coords[:,1]] = [255,0,0]   # 最黒点を赤
    if len(max_coords) > 0:
        vis_points[max_coords[:,0], max_coords[:,1]] = [0,255,0]   # 最白点を緑
    st.subheader("最も黒い点（赤）と最も白い点（緑）の位置")
    st.image(vis_points, channels="RGB")

    # --- ヒストグラム可視化 ---
    hist_vals, bins = np.histogram(gray.flatten(), bins=256, range=[0,256])
    fig, ax = plt.subplots()
    ax.plot(bins[:-1], hist_vals, color='black')
    ax.axvline(min_val, color='red', linestyle=':', label=f"min={min_val}")
    ax.axvline(max_val, color='green', linestyle=':', label=f"max={max_val}")
    ax.set_title("画素値ヒストグラム")
    ax.set_xlabel("画素値 (0=黒, 255=白)")
    ax.set_ylabel("画素数")
    ax.legend()
    st.pyplot(fig)

    # --- 二値化 ---
    thresh = (min_val + max_val) // 2
    st.write(f"二値化しきい値: {thresh}")
    binary = (gray > thresh).astype(np.uint8) * 255
    st.subheader("二値化画像（白黒二極化）")
    st.image(binary, clamp=True, channels="GRAY")

    # --- 三角形検出パラメータ ---
    st.sidebar.header("三角形検出パラメータ")
    min_area = st.sidebar.slider("三角形検出最小面積", 100, 5000, 400)
    approx_epsilon_factor = st.sidebar.slider("三角形近似精度（小さいほど厳密）", 0.01, 0.2, 0.04)

    # --- Cannyエッジ画像 ---
    canny1 = st.sidebar.slider("Cannyしきい値1", 1, 200, 50)
    canny2 = st.sidebar.slider("Cannyしきい値2", 1, 200, 150)
    edges = cv2.Canny(binary, canny1, canny2)
    st.subheader("Cannyエッジ画像")
    st.image(edges, clamp=True)

    # --- 三角形（支点）の検出 ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    triangle_mask = np.zeros_like(binary)
    triangle_table = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        epsilon = approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            triangles.append(approx)
            cv2.drawContours(triangle_mask, [approx], -1, 255, -1)
            pts = approx.reshape(-1, 2)
            triangle_table.append({
                "index": len(triangle_table)+1,
                "pt1_x": int(pts[0][0]),
                "pt1_y": int(pts[0][1]),
                "pt2_x": int(pts[1][0]),
                "pt2_y": int(pts[1][1]),
                "pt3_x": int(pts[2][0]),
                "pt3_y": int(pts[2][1]),
                "area": area
            })

    st.write(f"検出された三角形数: {len(triangles)}")

    # --- 三角形領域を除外した画像を生成 ---
    binary_no_tri = cv2.bitwise_and(binary, cv2.bitwise_not(triangle_mask))

    # --- 梁検出パラメータ ---
    st.sidebar.header("梁検出パラメータ")
    hough_thresh = st.sidebar.slider("HoughLinesP threshold", 10, 300, 100)
    min_line_length = st.sidebar.slider("minLineLength", 10, 300, 80)
    max_line_gap = st.sidebar.slider("maxLineGap", 1, 50, 10)
    horizontal_thresh_deg = st.sidebar.slider("水平判定の角度幅（度）", 1, 30, 8)
    y_thresh = st.sidebar.slider("梁グループ化 yしきい値", 2, 30, 8)

    # --- ハフ変換で直線検出（梁） ---
    lines = cv2.HoughLinesP(
        binary_no_tri, 1, np.pi / 180,
        threshold=hough_thresh, minLineLength=min_line_length, maxLineGap=max_line_gap
    )

    # --- 水平直線のみ抽出 ---
    horizontal_lines = []
    raw_lines_table = []
    line_index_map = {}
    if lines is not None and len(lines) > 0:
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            angle_deg = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if angle_deg < horizontal_thresh_deg or angle_deg > (180 - horizontal_thresh_deg):
                horizontal_lines.append(line)
                line_index_map[id(line)] = len(raw_lines_table) + 1
                if (x2-x1) != 0:
                    slope = (y2-y1)/(x2-x1)
                    intercept = y1 - slope*x1
                else:
                    slope = None
                    intercept = None
                raw_lines_table.append({
                    "index": len(raw_lines_table)+1,
                    "start_x": int(x1),
                    "start_y": int(y1),
                    "end_x": int(x2),
                    "end_y": int(y2),
                    "slope": slope,
                    "intercept": intercept,
                    "angle_deg": angle_deg
                })

    # --- 水平直線をグループ化（y座標＋x方向重なり） ---
    def group_lines_by_y_and_xoverlap(lines, y_thresh=8, x_overlap_ratio=0.2):
        if not lines:
            return []
        line_props = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            y_mean = np.mean([y1, y2])
            x_min, x_max = min(x1, x2), max(x1, x2)
            line_props.append({'line': line, 'y_mean': y_mean, 'x_min': x_min, 'x_max': x_max})
        used = [False] * len(lines)
        groups = []
        for i, prop in enumerate(line_props):
            if used[i]: continue
            group = [prop['line']]
            used[i] = True
            for j in range(i+1, len(line_props)):
                if used[j]: continue
                y_close = abs(line_props[j]['y_mean'] - prop['y_mean']) < y_thresh
                x_overlap = (min(prop['x_max'], line_props[j]['x_max']) - max(prop['x_min'], line_props[j]['x_min'])) > x_overlap_ratio*(prop['x_max']-prop['x_min'])
                if y_close and x_overlap:
                    group.append(line_props[j]['line'])
                    used[j] = True
            groups.append(group)
        return groups

    def fit_and_draw_horizontal_line_fitline(blank, group, color, thickness, index, line_index_map):
        pts = []
        raw_indices = []
        for line in group:
            x1, y1, x2, y2 = line[0]
            pts.append([x1, y1])
            pts.append([x2, y2])
            if id(line) in line_index_map:
                raw_indices.append(line_index_map[id(line)])
        pts = np.array(pts)
        if len(pts) < 2:
            return None
        fit = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = fit.flatten()
        height, width = blank.shape[:2]
        x_start, x_end = 0, width-1
        if abs(vx) < 1e-5:
            y_start = y_end = int(y0)
            slope = None
            intercept = None
        else:
            slope = vy / vx
            intercept = y0 - slope * x0
            y_start = int(slope * x_start + intercept)
            y_end = int(slope * x_end + intercept)
        cv2.line(blank, (x_start, y_start), (x_end, y_end), color, thickness)
        return {
            "index": index+1,
            "start_x": int(x_start),
            "start_y": int(y_start),
            "end_x": int(x_end),
            "end_y": int(y_end),
            "slope": float(slope) if slope is not None else None,
            "intercept": float(intercept) if intercept is not None else None,
            "group_size": len(group),
            "raw_indices": raw_indices
        }

    # --- 清書画像（水平梁＋三角形両方描画） ---
    height, width = image.shape[:2]
    blank = np.full((height, width, 3), 255, dtype=np.uint8)

    # 水平梁をグループ化して一本化して描画（青）＆パラメータ収集
    lines_table = []
    if horizontal_lines:
        groups = group_lines_by_y_and_xoverlap(
            horizontal_lines, y_thresh=y_thresh, x_overlap_ratio=0.2
        )
        for idx, group in enumerate(groups):
            line_info = fit_and_draw_horizontal_line_fitline(blank, group, (255,0,0), 3, idx, line_index_map)
            if line_info:
                lines_table.append(line_info)

    # 三角形を赤で描画
    for triangle in triangles:
        cv2.drawContours(blank, [triangle], -1, (0, 0, 255), -1)

    st.subheader("清書された構造図（一本化した水平梁＋三角形）")
    st.image(blank, caption="水平梁と三角形", channels="RGB")

    # --- テーブルおよびダウンロード ---
    if triangle_table:
        st.subheader("検出された三角形（支点）の一覧テーブル")
        df_tri = pd.DataFrame(triangle_table)
        st.dataframe(df_tri)
        csv_tri = df_tri.to_csv(index=False).encode('utf-8')
        st.download_button("三角形データCSVをダウンロード", csv_tri, file_name="triangles_table.csv", mime="text/csv")

    if raw_lines_table:
        st.subheader("元画像から検出された水平直線（RAW）の一覧テーブル")
        df_raw = pd.DataFrame(raw_lines_table)
        st.dataframe(df_raw)
        csv_raw = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button("RAW水平直線CSVをダウンロード", csv_raw, file_name="raw_horizontal_lines_table.csv", mime="text/csv")

    if lines_table:
        st.subheader("清書された水平直線（梁）の一覧テーブル（近似元情報付き）")
        df_clean = pd.DataFrame(lines_table)
        df_clean["近似元RAW直線index"] = df_clean["raw_indices"].apply(lambda x: ",".join(map(str, x)))
        df_clean = df_clean.drop("raw_indices", axis=1)
        st.dataframe(df_clean)
        csv_clean = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("清書水平直線CSV（近似元情報付き）をダウンロード", csv_clean, file_name="cleaned_horizontal_lines_table.csv", mime="text/csv")

    _, buf = cv2.imencode('.png', cv2.cvtColor(blank, cv2.COLOR_RGB2BGR))
    st.download_button("清書画像をダウンロード", buf.tobytes(), file_name="cleaned_structure_horizontal_and_triangles.png", mime="image/png")