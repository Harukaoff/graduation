import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile

st.title("📐 構造図から応力図を自動生成するツール")
st.markdown("1マス=1m（長さ）・1マス=5N（荷重）で換算。まず基準物エリアを指定してください。")

uploaded_file = st.file_uploader("📤 画像をアップロード", type=["jpg", "jpeg", "png"])

def line_thickness(gray, x1, y1, x2, y2):
    # 線の周辺の幅を調べる簡易版
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return 0
    mask = np.zeros_like(gray)
    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=3)  # 太めの線をマスク
    line_pixels = cv2.bitwise_and(gray, gray, mask=mask)
    count = cv2.countNonZero(line_pixels)
    return count / length

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img_path = tfile.name

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(img, caption="アップロード画像", use_column_width=True)

    st.markdown("### 基準物（定規・方眼）を囲む四角をドラッグで選択してください。")
    bbox = st.selectbox("基準物の領域を選択（最初は全画面を推奨）", options=["手動入力"])

    # 基準物領域をユーザーに入力させる（ここはStreamlitの座標入力で代用）
    st.markdown("基準物エリアの左上X座標")
    x_start = st.number_input("x_start", min_value=0, max_value=img.shape[1], value=0)
    st.markdown("基準物エリアの左上Y座標")
    y_start = st.number_input("y_start", min_value=0, max_value=img.shape[0], value=0)
    st.markdown("基準物エリアの幅")
    w = st.number_input("width", min_value=1, max_value=img.shape[1]-x_start, value=100)
    st.markdown("基準物エリアの高さ")
    h = st.number_input("height", min_value=1, max_value=img.shape[0]-y_start, value=100)

    # 基準物エリアの矩形を画像に表示
    img_show = img.copy()
    cv2.rectangle(img_show, (x_start, y_start), (x_start + w, y_start + h), (0, 255, 0), 3)
    st.image(img_show, caption="基準物エリア（緑枠）", use_column_width=True)

    # 基準物エリアからスケール計算（横方向の線の間隔＝1マスのピクセル数）
    scale_roi = gray[y_start:y_start+h, x_start:x_start+w]
    edges_roi = cv2.Canny(scale_roi, 50, 150)

    # 水平方向の線を探す（基準物の方眼線想定）
    lines_roi = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=5)
    if lines_roi is None:
        st.error("基準物の方眼線が検出できません。枠の位置を調整してください。")
        st.stop()

    # 水平線のY座標リスト抽出（方眼線を複数検出し距離からスケール計算）
    h_lines_y = []
    for line in lines_roi:
        x1_, y1_, x2_, y2_ = line[0]
        if abs(y2_ - y1_) < 5:  # 水平線のみ
            h_lines_y.append(y1_)

    h_lines_y = sorted(list(set(h_lines_y)))
    if len(h_lines_y) < 2:
        st.error("方眼の水平線が少なすぎます。")
        st.stop()

    # 線間のピクセル距離を計算し平均をスケール(px/マス)とする
    diffs = [j - i for i, j in zip(h_lines_y[:-1], h_lines_y[1:])]
    cell_size_px = np.mean(diffs)

    st.success(f"基準物の1マスの大きさ = {cell_size_px:.2f} ピクセル")

    # 梁・荷重の検出開始
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    beam_line = None
    loads = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 基準物エリア内の線は無視する
        if (x_start <= x1 <= x_start + w and y_start <= y1 <= y_start + h) or (x_start <= x2 <= x_start + w and y_start <= y2 <= y_start + h):
            continue

        # 梁検出（ほぼ水平かつ長い）
        if abs(y2 - y1) < 5 and abs(x2 - x1) > 150:
            thickness = line_thickness(gray, x1, y1, x2, y2)
            if thickness > 1.0:  # 太さチェック
                beam_line = (x1, y1, x2, y2)

        # 荷重検出（ほぼ垂直かつ適度な長さ）
        elif abs(x2 - x1) < 5 and 20 < abs(y2 - y1) < 80:
            thickness = line_thickness(gray, x1, y1, x2, y2)
            if thickness > 0.5:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                length_px = abs(y2 - y1)
                loads.append((cx, cy, length_px))

    if beam_line is None:
        st.error("梁が検出できませんでした。画像を確認してください。")
        st.stop()

    x1, y1, x2, y2 = beam_line
    beam_length_px = abs(x2 - x1)
    beam_length_m = beam_length_px / cell_size_px  # ピクセルをマスに変換＝m

    st.success(f"検出された梁の長さ：{beam_length_m:.2f} m (ピクセル長さ: {beam_length_px})")

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, label="検出梁")
    ax.add_patch(plt.Rectangle((x_start, y_start), w, h, edgecolor='lime', facecolor='none', linewidth=2, label="基準物エリア"))
    ax.legend()
    ax.set_title("検出された構造図と基準物")
    st.pyplot(fig)

    # せん断力・モーメント計算
    x_vals = np.linspace(0, beam_length_m, int(beam_length_m * 10) + 1)
    V = np.zeros_like(x_vals, dtype=float)
    M = np.zeros_like(x_vals, dtype=float)

    for (cx, cy, length_px) in loads:
        # 荷重位置(m)
        load_pos_px = cx - min(x1, x2)
        load_pos_m = load_pos_px / cell_size_px

        # 荷重値 (N) = 1マス=5Nなので、長さ(px)をマスに変換して5N掛ける
        load_val = (length_px / cell_size_px) * 5

        st.write(f"🔻 荷重検出: 位置={load_pos_m:.2f} m, 強さ={load_val:.2f} N")

        # せん断力・モーメントに反映
        V[x_vals >= load_pos_m] -= load_val
        M[x_vals >= load_pos_m] -= load_val * (x_vals[x_vals >= load_pos_m] - load_pos_m)

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(x_vals, V, label="Shear Force [N]", color='red')
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.set_ylabel("Shear Force [N]")
    ax1.set_title("せん断力図")

    ax2.plot(x_vals, M, label="Bending Moment [Nm]", color='blue')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_xlabel("梁の位置 [m]")
    ax2.set_ylabel("Bending Moment [Nm]")
    ax2.set_title("曲げモーメント図")

    st.pyplot(fig2)
