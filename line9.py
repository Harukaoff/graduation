import streamlit as st
import numpy as np
import cv2
from PIL import Image
from collections import Counter
import math

# ヘルパー関数: 2点間の距離
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ヘルパー関数: 3点からなす角を計算 (度数)
def get_angle(p1, p_center, p2):
    vec1 = (p1[0] - p_center[0], p1[1] - p_center[1])
    vec2 = (p2[0] - p_center[0], p2[1] - p_center[1])
    
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    magnitude2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    angle_rad = math.acos(min(max(dot_product / (magnitude1 * magnitude2), -1.0), 1.0))
    return math.degrees(angle_rad)

# ヘルパー関数: 線分の角度 (0-360度)
def get_line_orientation_angle(p1, p2):
    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    if angle < 0:
        angle += 360
    return angle

def detect_structures(image: Image.Image,
                      # 画像縮小パラメータ
                      resize_scale: float,
                      # 前処理パラメータ
                      blur_kernel: int,
                      threshold_type: str,
                      block_size: int, C_value: int,
                      # Cannyエッジ検出パラメータ
                      canny_lower: int, canny_upper: int,
                      # モルフォロジー変換パラメータ
                      morph_kernel_size: int,
                      # 三角形検出パラメータ
                      min_triangle_area: int,
                      approx_epsilon_factor: float,
                      # 直線検出パラメータ (Hough)
                      hough_threshold: int,
                      min_line_length_hough: int,
                      max_line_gap_hough: int,
                      line_angle_tolerance: int, # 水平・垂直判定の許容誤差
                      # 矢印検出パラメータ
                      arrow_head_max_len_ratio: float, # 本体線に対する矢羽根の最大長さ比率
                      arrow_head_angle_min: int, # 矢羽根の最小開き角度 (度)
                      arrow_head_angle_max: int, # 矢羽根の最大開き角度 (度)
                      arrow_connection_tolerance: int # 矢羽根と本体線の接続許容誤差
                      ):
    """
    入力画像から三角形、矢印、直線を検出し、検出結果を描画した画像を返します。
    """
    # PIL画像をOpenCV形式 (BGR) に変換
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # --- 画像の縮小 ---
    if resize_scale < 1.0:
        new_width = int(image_bgr.shape[1] * resize_scale)
        new_height = int(image_bgr.shape[0] * resize_scale)
        image_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 描画用の出力画像は最初からBGRでコピーしておく
    output_image = np.copy(image_bgr)

    # グレースケール画像を作成
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- 1. ノイズ除去（ガウシアンブラー）---
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred_image = cv2.GaussianBlur(gray_image, (blur_kernel, blur_kernel), 0)

    # --- 2. 二値化 ---
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    
    if threshold_type == "Otsu":
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = cv2.bitwise_not(binary_image)
    elif threshold_type == "Adaptive Mean":
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, block_size, C_value)
    elif threshold_type == "Adaptive Gaussian":
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, block_size, C_value)
    else: # Simple Binary (固定閾値)
        _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    # --- 3. Cannyエッジ検出 ---
    edges = cv2.Canny(binary_image, canny_lower, canny_upper)

    # --- 4. モルフォロジー変換（クロージング）---
    morph_kernel_size = morph_kernel_size if morph_kernel_size % 2 == 1 else morph_kernel_size + 1
    if morph_kernel_size > 1:
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    detected_elements = []
    used_line_indices = set() # 矢印の構成要素として使われたHough線分のインデックス

    # --- 5. 輪郭検出に基づく三角形の検出 ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 縮小された画像上での最小面積を調整
    scaled_min_triangle_area = min_triangle_area * (resize_scale ** 2)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < scaled_min_triangle_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        if num_vertices == 3:
            if cv2.isContourConvex(approx):
                shape_name = "三角形"
                color = (0, 255, 0) # 緑
                cv2.drawContours(output_image, [approx], 0, color, 4)
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(output_image, shape_name, (cX - 30, cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    detected_elements.append(shape_name)

    # --- 6. 直線と矢印の検出 ---
    # Hough変換で全ての線分を検出 (後で矢印と普通の直線に分類)
    lines_hough = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length_hough,
        maxLineGap=max_line_gap_hough
    )

    if lines_hough is not None:
        lines_list = [tuple(line[0]) for line in lines_hough] # タプルに変換してイミュータブルにする

        # 矢印検出
        # 処理負荷軽減のため、線分の端点を格納し、近傍探索を試みることも可能だが、
        # まずはパラメータ調整と画像縮小で様子を見る。
        
        for i, (lx1, ly1, lx2, ly2) in enumerate(lines_list):
            if i in used_line_indices:
                continue

            main_p1 = (lx1, ly1)
            main_p2 = (lx2, ly2)
            main_len = dist(main_p1, main_p2)

            if main_len < min_line_length_hough:
                continue

            # 矢印の先端になりうる場所をチェック (main_p1 と main_p2)
            for tip_candidate_idx in [0, 1]:
                tip_x, tip_y = main_p1 if tip_candidate_idx == 0 else main_p2
                
                main_vec_end_x, main_vec_end_y = main_p2 if tip_candidate_idx == 0 else main_p1
                
                # 矢印の先端 (tip_x, tip_y) から本体線が伸びる方向の角度
                main_vec_angle = get_line_orientation_angle((tip_x, tip_y), (main_vec_end_x, main_vec_end_y))
                
                arrow_head_lines_with_info = [] # (線分座標, 角度, インデックス) を格納

                # 矢羽根候補を探す
                for j, (hx1, hy1, hx2, hy2) in enumerate(lines_list):
                    if i == j or j in used_line_indices:
                        continue

                    head_len = dist((hx1, hy1), (hx2, hy2))
                    if head_len > main_len * arrow_head_max_len_ratio or head_len < 5:
                        continue

                    is_connected = False
                    head_other_end = (0,0)
                    head_start_point = (0,0)

                    if dist((hx1, hy1), (tip_x, tip_y)) < arrow_connection_tolerance:
                        is_connected = True
                        head_start_point = (hx1, hy1)
                        head_other_end = (hx2, hy2)
                    elif dist((hx2, hy2), (tip_x, tip_y)) < arrow_connection_tolerance:
                        is_connected = True
                        head_start_point = (hx2, hy2)
                        head_other_end = (hx1, hy1)
                    
                    if is_connected:
                        head_angle = get_line_orientation_angle(head_start_point, head_other_end)

                        angle_diff = abs(main_vec_angle - head_angle)
                        angle_diff = min(angle_diff, 360 - angle_diff)

                        if arrow_head_angle_min < angle_diff < arrow_head_angle_max:
                            arrow_head_lines_with_info.append(((hx1, hy1, hx2, hy2), head_angle, j))
                
                # 矢羽根が2本以上見つかったら矢印と判定
                if len(arrow_head_lines_with_info) >= 2:
                    # 2本の矢羽根を選定 (最も条件に合うものを優先するなど、より洗練させることも可能)
                    # ここでは、単純に最初の2本を使用する
                    ah1_coords, angle1_raw, ah1_idx = arrow_head_lines_with_info[0]
                    ah2_coords, angle2_raw, ah2_idx = arrow_head_lines_with_info[1]

                    # 対称性チェック
                    rel_angle1 = (angle1_raw - main_vec_angle + 360) % 360
                    rel_angle2 = (angle2_raw - main_vec_angle + 360) % 360
                    
                    angle_span = abs(rel_angle1 - rel_angle2)
                    if not (100 < angle_span < 260): # 180±60度程度の範囲
                        continue

                    # 矢印として検出！
                    cv2.line(output_image, main_p1, main_p2, (255, 0, 255), 3) # 本体線（マゼンタ）
                    cv2.line(output_image, (ah1_coords[0], ah1_coords[1]), (ah1_coords[2], ah1_coords[3]), (255, 0, 255), 3)
                    cv2.line(output_image, (ah2_coords[0], ah2_coords[1]), (ah2_coords[2], ah2_coords[3]), (255, 0, 255), 3)
                    
                    cv2.putText(output_image, "Arrow", (tip_x, tip_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    detected_elements.append("矢印")
                    used_line_indices.add(i) # 本体線を使用済み
                    used_line_indices.add(ah1_idx) # 矢羽根1を使用済み
                    used_line_indices.add(ah2_idx) # 矢羽根2を使用済み
                    
                    break # この本体線からの矢印検出は終了し、次の本体線へ

        # 矢印として使われなかった線分を直線として描画
        for i, (x1, y1, x2, y2) in enumerate(lines_list):
            if i in used_line_indices:
                continue

            line_length = dist((x1,y1), (x2,y2))
            if line_length < min_line_length_hough:
                continue

            angle_deg = get_line_orientation_angle((x1,y1), (x2,y2)) % 180 # 0-180度に正規化

            is_horizontal = (angle_deg < line_angle_tolerance) or (abs(angle_deg - 180) < line_angle_tolerance)
            is_vertical = (abs(angle_deg - 90) < line_angle_tolerance)
            
            line_type = "直線"
            color = (255, 165, 0) # オレンジ色 (デフォルトの斜線)

            if is_horizontal:
                line_type = "水平線"
                color = (255, 0, 0) # 青
            elif is_vertical:
                line_type = "垂直線"
                color = (0, 0, 255) # 赤
            
            cv2.line(output_image, (x1, y1), (x2, y2), color, 2)
            detected_elements.append(line_type)

    # 結果画像をPIL形式に戻して返す
    return Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)), detected_elements

# --- Streamlit UIの構築 ---
st.set_page_config(layout="wide", page_title="手書き構造物検出アプリ")

st.title("手書き対応：構造物要素検出アプリ (三角形・矢印・直線)")
st.write("画像をアップロードし、左のサイドバーでパラメータを調整して、**三角形、矢印、直線**の検出を試してください。")
st.write("特に**手書き画像**の場合、[1. ノイズ除去]、[2. 二値化]、[4. 線分の補強]、そして**[画像の縮小率]** のパラメータ調整が重要です。")

uploaded_file = st.file_uploader("ここに画像をドラッグ＆ドロップするか、ファイルを選択してください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image_pil = Image.open(uploaded_file)
    st.subheader("アップロードされた画像:")
    st.image(input_image_pil, caption="元の画像", use_column_width=True)

    st.sidebar.header("前処理と検出パラメータ")

    # --- 画像縮小 ---
    st.sidebar.markdown("### 0. 画像の縮小 (高速化のため)")
    resize_scale = st.sidebar.slider(
        "画像縮小率", min_value=0.1, max_value=1.0, value=0.5, step=0.1, format="%.1f",
        help="処理前に画像を縮小します。値を小さくすると高速化しますが、検出精度が落ちる可能性があります。"
    )


    # --- 1. ノイズ除去 ---
    st.sidebar.markdown("### 1. ノイズ除去（ガウシアンブラー）")
    blur_kernel_size = st.sidebar.slider(
        "ガウシアンブラー カーネルサイズ", min_value=1, max_value=21, value=5, step=2, # 奇数のみ
        help="値を大きくすると、より強くぼかされノイズが除去されます。線の太さもわずかに変わります。"
    )

    # --- 2. 二値化 ---
    st.sidebar.markdown("### 2. 二値化（線をはっきりさせる）")
    threshold_method = st.sidebar.selectbox(
        "二値化方法",
        ("Adaptive Gaussian", "Adaptive Mean", "Otsu", "Simple Binary"),
        index=0, # デフォルトはAdaptive Gaussian
        help="手書き画像には 'Adaptive Gaussian' や 'Adaptive Mean' がおすすめです。Otsuはグローバル閾値です。"
    )
    
    block_size_val = 11
    C_value_val = 2
    if threshold_method in ["Adaptive Mean", "Adaptive Gaussian"]:
        block_size_val = st.sidebar.slider(
            "適応的閾値のブロックサイズ", min_value=3, max_value=51, value=11, step=2, # 奇数のみ
            help="周囲の平均を計算する領域のサイズです。線の太さや背景のムラに応じて調整。"
        )
        C_value_val = st.sidebar.slider(
            "適応的閾値のC値", min_value=-10, max_value=10, value=2, step=1,
            help="平均値から引かれる定数。明るい背景に暗い線の場合、正の値が有効です。"
        )

    # --- 3. Cannyエッジ検出 ---
    st.sidebar.markdown("### 3. Cannyエッジ検出")
    canny_lower_threshold = st.sidebar.slider("Canny下限閾値", 0, 255, 50, 1,
        help="低い値ほど細かいエッジも検出します。ノイズが多くなる可能性も。"
    )
    canny_upper_threshold = st.sidebar.slider("Canny上限閾値", 0, 255, 150, 1,
        help="高い値ほど強いエッジのみ検出します。線の途切れが発生する可能性も。"
    )

    # --- 4. 線分の補強（モルフォロジー変換） ---
    st.sidebar.markdown("### 4. 線分の補強（クロージング）")
    morph_kernel_size_val = st.sidebar.slider(
        "クロージング カーネルサイズ", min_value=1, max_value=7, value=1, step=2, # 奇数のみ (1は実質オフ)
        help="二値化後の途切れた線を繋げたり、小さな穴を埋めたりします。値を大きくすると効果が強まります。"
    )

    # --- 5. 三角形検出の最終調整 ---
    st.sidebar.markdown("### 5. 三角形検出の調整")
    min_triangle_area_val = st.sidebar.slider(
        "最小三角形面積 (px^2)", min_value=10, max_value=5000, value=500, step=10,
        help="これより小さい面積の三角形は無視されます。ノイズ除去に役立ちます。**画像の縮小率に合わせて調整してください。**"
    )
    approx_epsilon_val = st.sidebar.slider(
        "輪郭近似の精度 (0.01 - 0.20)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f",
        help="値が小さいほど元の輪郭に忠実になりますが、手書きの揺らぎを拾いすぎることがあります。値を大きくするとより単純な形状に近似され、手書きの不正確さを吸収できます。"
    )

    # --- 6. 直線検出 (Hough Lines P) ---
    st.sidebar.markdown("### 6. 直線検出 (Hough Lines P)")
    st.sidebar.info("**パフォーマンス改善の鍵**: ここでのHough変換パラメータは、検出速度に大きな影響を与えます。適切に調整してください。")
    hough_threshold_val = st.sidebar.slider(
        "Hough変換の最小交点数", min_value=1, max_value=200, value=10, step=1,
        help="この値を**大きくする**と、検出される線分の数が減り、処理が高速化します。ただし、弱い線は見逃されます。"
    )
    min_line_length_hough_val = st.sidebar.slider(
        "線分の最小長さ", min_value=1, max_value=300, value=10, step=1,
        help="これより短い線分はHough変換で無視されます。**この値を上げる**と、処理が高速化します。"
    )
    max_line_gap_hough_val = st.sidebar.slider(
        "最大ギャップ", min_value=0, max_value=100, value=8, step=1,
        help="同一直線上と見なされる線分間の最大ギャップです。この値を**小さくする**と、線分の数が増えすぎずに済みます。"
    )
    line_angle_tolerance_val = st.sidebar.slider(
        "直線判定角度許容誤差 (度)", min_value=1, max_value=10, value=5, step=1,
        help="水平線・垂直線を判定する際の角度の許容誤差です。"
    )

    # --- 7. 矢印検出 (Hough Lines P と幾何分析) ---
    st.sidebar.markdown("### 7. 矢印検出 (Hough Lines P と幾何分析)")
    arrow_head_max_len_ratio_val = st.sidebar.slider(
        "矢羽根の本体線に対する最大長比率", min_value=0.1, max_value=0.8, value=0.4, step=0.05, format="%.2f",
        help="矢羽根の線が本体線に対してどのくらいの長さまで許容されるか。**画像の縮小率に合わせて調整してください。**"
    )
    arrow_head_angle_min_val = st.sidebar.slider(
        "矢羽根の最小開き角度 (度)", min_value=10, max_value=60, value=20, step=1,
        help="矢羽根の線が本体線から開く最小角度。"
    )
    arrow_head_angle_max_val = st.sidebar.slider(
        "矢羽根の最大開き角度 (度)", min_value=30, max_value=90, value=50, step=1,
        help="矢羽根の線が本体線から開く最大角度。"
    )
    arrow_connection_tolerance_val = st.sidebar.slider(
        "矢羽根接続許容誤差 (px)", min_value=1, max_value=20, value=5, step=1,
        help="矢羽根と本体線が接続しているとみなす端点間の最大距離。**画像の縮小率に合わせて調整してください。**"
    )


    st.write("---")
    st.write("構造物要素を検出中...")

    # 全検出処理を実行
    processed_image_pil, detected_elements_list = detect_structures(
        input_image_pil,
        resize_scale, # 新しいパラメータ
        blur_kernel_size,
        threshold_method,
        block_size_val, C_value_val,
        canny_lower_threshold,
        canny_upper_threshold,
        morph_kernel_size_val,
        min_triangle_area_val,
        approx_epsilon_val,
        hough_threshold_val,
        min_line_length_hough_val,
        max_line_gap_hough_val,
        line_angle_tolerance_val,
        arrow_head_max_len_ratio_val,
        arrow_head_angle_min_val,
        arrow_head_angle_max_val,
        arrow_connection_tolerance_val
    )

    st.subheader("検出結果:")
    st.image(processed_image_pil, caption="検出された構造物要素", use_column_width=True)

    # 検出された要素のリスト表示
    if detected_elements_list:
        st.subheader("検出された要素の概要:")
        element_counts = Counter(detected_elements_list)
        for element, count in element_counts.items():
            st.write(f"- {element}: {count}個")
    else:
        st.write("指定された図形要素は検出されませんでした。サイドバーのパラメータを調整してみてください。")

st.markdown("---")
st.write("© 2025 手書き構造物検出アプリ")