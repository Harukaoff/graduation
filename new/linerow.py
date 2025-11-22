import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.title("構造図の清書（水平梁のみ一本化・三角形除外）")

uploaded_file = st.file_uploader(
    "構造図画像をアップロードしてください",
    type=["png", "jpg", "jpeg"],
    key="main_uploader"
)

if uploaded_file is not None:
    # 読み込み・前処理
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 二値化（白背景＋黒線に）
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    st.subheader("二値化画像（反転）")
    st.image(binary, clamp=True)

    # 三角形（支点）の検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    triangle_mask = np.zeros_like(binary)
    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            area = cv2.contourArea(cnt)
            if area > 100:
                triangles.append(approx)
                # 三角形をマスク
                cv2.drawContours(triangle_mask, [approx], -1, 255, -1)

    # 三角形領域を除外した画像を生成
    binary_no_tri = cv2.bitwise_and(binary, cv2.bitwise_not(triangle_mask))

    # 梁（直線）をハフ変換で検出（除外済み画像で）
    lines = cv2.HoughLinesP(binary_no_tri, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)

    # 水平直線のみ抽出（角度でフィルタ）
    horizontal_lines = []
    horizontal_thresh_deg = 10  # 水平判定±10度
    raw_lines_table = []
    if lines is not None and len(lines) > 0:
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            angle_deg = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            # 水平（0度または180度近辺）のみ
            if angle_deg < horizontal_thresh_deg or angle_deg > (180 - horizontal_thresh_deg):
                horizontal_lines.append(line)
                # 傾き・切片
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

    # 水平直線をグループ化
    def group_lines_by_y(lines, y_thresh=15):
        # y座標が近い直線同士を同じグループに
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

    def fit_and_draw_horizontal_line(blank, group, color, thickness, index):
        # x座標の最小・最大を端点に使う
        xs = []
        ys = []
        for line in group:
            x1, y1, x2, y2 = line[0]
            xs += [x1, x2]
            ys += [y1, y2]
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
            "group_size": len(group)
        }

    # 清書用の白背景カラー画像
    height, width = image_bgr.shape[:2]
    blank = np.full((height, width, 3), 255, dtype=np.uint8)

    # 水平梁をグループ化して一本化して描画（青）＆パラメータ収集
    lines_table = []
    if horizontal_lines:
        groups = group_lines_by_y(horizontal_lines, y_thresh=15)
        for idx, group in enumerate(groups):
            line_info = fit_and_draw_horizontal_line(blank, group, (255,0,0), 3, idx)
            if line_info:
                lines_table.append(line_info)

    # 支点（三角形）を赤で描画
    for triangle in triangles:
        cv2.drawContours(blank, [triangle], -1, (0, 0, 255), -1)  # 赤

    # 表示
    st.subheader("清書された構造図（水平方向のみ・三角形除外）")
    st.image(blank, channels="BGR")

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

    # 清書水平直線テーブル
    if lines_table:
        st.subheader("清書された水平直線（梁）の一覧テーブル")
        df_clean = pd.DataFrame(lines_table)
        st.dataframe(df_clean)
        csv_clean = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            "清書水平直線CSVをダウンロード",
            csv_clean,
            file_name="cleaned_horizontal_lines_table.csv",
            mime="text/csv"
        )

    # ダウンロードボタン（画像）
    _, buf = cv2.imencode('.png', blank)
    st.download_button(
        "清書画像をダウンロード",
        buf.tobytes(),
        file_name="cleaned_structure_horizontal.png",
        mime="image/png"
    )