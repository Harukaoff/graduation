import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.title("方眼紙付き構造図から構造線とスケールを認識")

uploaded_file = st.file_uploader("構造図画像をアップロード", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 方眼紙の色（薄い青）→ HSVでフィルタ（青系）
    lower_grid = np.array([90, 30, 150])   # 色味は画像に合わせて微調整してね
    upper_grid = np.array([130, 255, 255])
    grid_mask = cv2.inRange(hsv, lower_grid, upper_grid)

    # 構造線（黒）→ HSVでフィルタ（黒〜グレー）
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    structure_mask = cv2.inRange(hsv, lower_black, upper_black)

    # エッジ検出
    edges_grid = cv2.Canny(grid_mask, 50, 150)
    edges_structure = cv2.Canny(structure_mask, 50, 150)

    # 線分検出（方眼紙）
    grid_lines = cv2.HoughLinesP(edges_grid, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=5)

    # 線分検出（構造体）
    structure_lines = cv2.HoughLinesP(edges_structure, 1, np.pi / 180, threshold=80, minLineLength=30, maxLineGap=10)

    # 方眼からスケール推定（1マス1mmと仮定）
    grid_spacing_px = None
    if grid_lines is not None:
        vertical_lines = [line for line in grid_lines if abs(line[0][0] - line[0][2]) < 5]
        if len(vertical_lines) > 1:
            # 複数の縦線から間隔を推定
            x_coords = sorted([line[0][0] for line in vertical_lines])
            spacings = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
            grid_spacing_px = np.median(spacings)  # 1mmに相当するピクセル数

    scale_mm_per_pixel = 1 / grid_spacing_px if grid_spacing_px else None

    # 結果描画
    result_img = img.copy()
    if structure_lines is not None:
        for line in structure_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="構造線を検出した画像", use_column_width=True)

    if scale_mm_per_pixel:
        st.success(f"スケール認識成功！ 1ピクセル ≒ {scale_mm_per_pixel:.3f} mm")
    else:
        st.warning("スケール認識に失敗しました。方眼線の数やコントラストを確認してください。")

    # デバッグ表示（必要なら）
    # st.image(grid_mask, caption="方眼マスク", use_column_width=True)
    # st.image(structure_mask, caption="構造線マスク", use_column_width=True)
