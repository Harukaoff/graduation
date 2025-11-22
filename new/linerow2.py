import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.title("構造図の清書（水平梁のみ一本化・三角形検出・両方表示・パラメータ調整）")

uploaded_file = st.file_uploader(
    "構造図画像をアップロードしてください",
    type=["png", "jpg", "jpeg"],
    key="main_uploader"
)

if uploaded_file is not None:
    min_area = st.slider("三角形検出最小面積", 100, 5000, 500)
    approx_epsilon_factor = st.slider("近似精度（小さいほど厳密）", 0.01, 0.2, 0.04)
    block_size = st.slider("adaptiveThreshold blockSize", 3, 61, 21, step=2)
    c_value = st.slider("adaptiveThreshold C", 0, 15, 6)
    canny1 = st.slider("Cannyしきい値1", 1, 200, 50)
    canny2 = st.slider("Cannyしきい値2", 1, 200, 150)
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # adaptiveThreshold→Canny
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_value)
    edges = cv2.Canny(binary, canny1, canny2)

    st.subheader("adaptiveThreshold+エッジ画像")
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

    # --- 梁（直線）をハフ変換で検出（除外済み画像で） ---
    lines = cv2.HoughLinesP(binary_no_tri, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)

    # --- 水平直線のみ抽出 ---
    horizontal_lines = []
    horizontal_thresh_deg = 10  # 水平判定±10度
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

    # --- 水平直線をグループ化 ---
    def group_lines_by_y(lines, y_thresh=15):
        if not lines: return []
        ys = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ys.append((y1, y2))
        avg_ys = [np.mean(y_pair) for y_pair in ys]
        used = [False]*len(lines)
        groups = []
        for i, y in enumerate(avg_ys):
            if used[i]: continue
            group = [lines[i]]
            used[i] = True
            for j in range(i+1, len(lines)):
                if not used[j] and abs(avg_ys[j] - y) < y_thresh:
                    group.append(lines[j])
                    used[j] = True
            groups.append(group)
        return groups

    def fit_and_draw_horizontal_line(blank, group, color, thickness, index, line_index_map):
        xs = []
        ys = []
        raw_indices = []
        for line in group:
            x1, y1, x2, y2 = line[0]
            xs += [x1, x2]
            ys += [y1, y2]
            if id(line) in line_index_map:
                raw_indices.append(line_index_map[id(line)])
        min_x, max_x = min(xs), max(xs)
        avg_y = int(np.mean(ys))
        cv2.line(blank, (min_x, avg_y), (max_x, avg_y), color, thickness)
        return {
            "index": index+1,
            "start_x": int(min_x),
            "start_y": int(avg_y),
            "end_x": int(max_x),
            "end_y": int(avg_y),
            "slope": 0.0,
            "intercept": float(avg_y),
            "group_size": len(group),
            "raw_indices": raw_indices
        }

    # --- 清書画像（水平梁＋三角形両方描画） ---
    height, width = image_rgb.shape[:2]
    blank = np.full((height, width, 3), 255, dtype=np.uint8)

    # 水平梁をグループ化して一本化して描画（青）＆パラメータ収集
    lines_table = []
    if horizontal_lines:
        groups = group_lines_by_y(horizontal_lines, y_thresh=15)
        for idx, group in enumerate(groups):
            line_info = fit_and_draw_horizontal_line(blank, group, (255,0,0), 3, idx, line_index_map)
            if line_info:
                lines_table.append(line_info)

    # 三角形を赤で描画
    for triangle in triangles:
        cv2.drawContours(blank, [triangle], -1, (0, 0, 255), -1)

    st.subheader("清書された構造図（水平方向のみ・三角形も表示）")
    st.image(blank, caption="水平梁と三角形", channels="RGB")

    # 三角形テーブル
    if triangle_table:
        st.subheader("検出された三角形（支点）の一覧テーブル")
        df_tri = pd.DataFrame(triangle_table)
        st.dataframe(df_tri)
        csv_tri = df_tri.to_csv(index=False).encode('utf-8')
        st.download_button(
            "三角形データCSVをダウンロード",
            csv_tri,
            file_name="triangles_table.csv",
            mime="text/csv"
        )

    # RAW水平直線テーブル
    if raw_lines_table:
        st.subheader("元画像から検出された水平直線（RAW）の一覧テーブル")
        df_raw = pd.DataFrame(raw_lines_table)
        st.dataframe(df_raw)
        csv_raw = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            "RAW水平直線CSVをダウンロード",
            csv_raw,
            file_name="raw_horizontal_lines_table.csv",
            mime="text/csv"
        )

    # 清書水平直線テーブル（どのRAW直線を近似したかも表示）
    if lines_table:
        st.subheader("清書された水平直線（梁）の一覧テーブル（近似元情報付き）")
        df_clean = pd.DataFrame(lines_table)
        df_clean["近似元RAW直線index"] = df_clean["raw_indices"].apply(lambda x: ",".join(map(str, x)))
        df_clean = df_clean.drop("raw_indices", axis=1)
        st.dataframe(df_clean)
        csv_clean = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            "清書水平直線CSV（近似元情報付き）をダウンロード",
            csv_clean,
            file_name="cleaned_horizontal_lines_table.csv",
            mime="text/csv"
        )

    # ダウンロードボタン（画像）
    _, buf = cv2.imencode('.png', cv2.cvtColor(blank, cv2.COLOR_RGB2BGR))
    st.download_button(
        "清書画像をダウンロード",
        buf.tobytes(),
        file_name="cleaned_structure_horizontal_and_triangles.png",
        mime="image/png"
    )