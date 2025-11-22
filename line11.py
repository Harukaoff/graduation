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
                      # 全体の検出感度
                      detection_sensitivity: float, # 0.1 (低感度) - 1.0 (高感度)
                      # 前処理パラメータ (一部は感度で調整)
                      blur_kernel: int,
                      threshold_type: str,
                      block_size: int, C_value: int,
                      # Cannyエッジ検出パラメータ (一部は感度で調整)
                      canny_lower_base: int, canny_upper_base: int,
                      # モルフォロジー変換パラメータ
                      morph_kernel_size: int,
                      # Hough変換の個別調整パラメータ
                      max_line_gap_hough_manual: int, # 手動で調整可能な最大ギャップ
                      # 矢印検出固有のパラメータ (一部は感度で調整)
                      arrow_head_angle_min: int,
                      arrow_head_angle_max: int,
                      ):
    """
    入力画像から三角形、矢印、直線を検出し、検出結果を描画した画像を返します。
    検出感度スライダーにより、主要なパラメータを自動調整します。
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

    # --- 検出感度に基づくパラメータ調整 ---
    # 感度が低いほど、より厳密な検出（ノイズを減らす、小さいものを無視する）
    # 感度が高いほど、より緩やかな検出（小さいものも拾う、少しノイズが増える可能性）

    # Canny閾値
    # 感度が高いほど、閾値の範囲が広がる (より多くのエッジを拾う)
    canny_lower = int(canny_lower_base * detection_sensitivity)
    # canny_upper は感度が高いほど低い値に設定し、より強いエッジに集中 (ハイコントラストなエッジを優先)
    # detection_sensitivity が1.0に近づくほど (1.0 - detection_sensitivity) が0に近づくため、
    # canny_upper は canny_upper_base の値に近づく。
    # detection_sensitivity が0.1に近づくほど (1.0 - detection_sensitivity) が0.9に近づくため、
    # canny_upper は canny_upper_base * 1.9 に近づく (つまり、高い値になり、弱いエッジも拾う)
    canny_upper = int(canny_upper_base * (1.0 + (1.0 - detection_sensitivity))) 
    if canny_lower >= canny_upper: # 閾値が逆転しないように調整
        canny_upper = canny_lower + 1 
    
    # Hough変換パラメータ
    # 感度が低いほど、より多くの投票が必要 (hough_threshold を高く)
    hough_threshold = int(st.session_state.base_hough_threshold / detection_sensitivity) 
    # 感度が低いほど、より長い線分が必要 (min_line_length_hough を高く)
    min_line_length_hough = int(st.session_state.base_min_line_length_hough / detection_sensitivity)
    # max_line_gap_hough は手動調整を優先するため、ここでは直接調整しない

    # 三角形最小面積 (縮小率も考慮)
    min_triangle_area = int(st.session_state.base_min_triangle_area / detection_sensitivity) * (resize_scale ** 2)
    # 輪郭近似の精度 (感度が高いほど、より細かく近似 -> 小さい値)
    approx_epsilon_factor = st.session_state.base_approx_epsilon_factor / detection_sensitivity

    # 矢羽根の最大長比率 (感度が高いほど、矢羽根が本体に対して長めでも許容 -> 高い値)
    arrow_head_max_len_ratio = st.session_state.base_arrow_head_max_len_ratio * detection_sensitivity
    # 矢羽根接続許容誤差 (感度が高いほど、接続が緩やかでも許容 -> 高い値)
    arrow_connection_tolerance = int(st.session_state.base_arrow_connection_tolerance * detection_sensitivity)
    
    # 直線判定角度許容誤差 (感度が高いほど、直線判定が緩やか -> 高い値)
    line_angle_tolerance = int(st.session_state.base_line_angle_tolerance * detection_sensitivity)


    # --- 1. ノイズ除去（ガウシアンブラー）---
    # カーネルサイズは奇数のみ
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred_image = cv2.GaussianBlur(gray_image, (blur_kernel, blur_kernel), 0)

    # --- 2. 二値化 ---
    # ブロックサイズは奇数のみ
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    
    if threshold_type == "Otsu":
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_image = cv2.bitwise_not(binary_image) # 線を白にするために反転
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
    # カーネルサイズは奇数のみ (1は実質オフ)
    morph_kernel_size = morph_kernel_size if morph_kernel_size % 2 == 1 else morph_kernel_size + 1
    if morph_kernel_size > 1:
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    detected_elements = []
    used_line_indices = set() # 矢印の構成要素として使われたHough線分のインデックス

    # --- 5. 輪郭検出に基づく三角形の検出 ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_triangle_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)
        
        if num_vertices == 3:
            # 凸性チェック (より厳密な三角形に)
            if cv2.isContourConvex(approx):
                shape_name = "三角形"
                color = (0, 255, 0) # 緑
                cv2.drawContours(output_image, [approx], 0, color, 4)
                
                # 検出された三角形の中心にテキストを描画
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
        rho=1, # 距離分解能 (ピクセル単位)
        theta=np.pi / 180, # 角度分解能 (ラジアン単位, 1度)
        threshold=hough_threshold, # 投票数の閾値。この値より多くの投票が集まった直線のみを検出
        minLineLength=min_line_length_hough, # 検出する線分の最小長さ
        maxLineGap=max_line_gap_hough_manual # 同一直線上と見なされる線分間の最大ギャップ (手動で調整)
    )

    if lines_hough is not None:
        lines_list = [tuple(line[0]) for line in lines_hough] # タプルに変換してイミュータブルにする

        # 矢印検出
        for i, (lx1, ly1, lx2, ly2) in enumerate(lines_list):
            if i in used_line_indices:
                continue # 既に使われた線はスキップ

            main_p1 = (lx1, ly1)
            main_p2 = (lx2, ly2)
            main_len = dist(main_p1, main_p2)

            # 本体線が短すぎる場合はスキップ (Hough変換のmin_line_length_houghと連動)
            if main_len < min_line_length_hough:
                continue

            # 矢印の先端になりうる場所をチェック (main_p1 と main_p2)
            for tip_candidate_idx in [0, 1]: # 0はmain_p1が先端、1はmain_p2が先端
                tip_x, tip_y = main_p1 if tip_candidate_idx == 0 else main_p2
                
                # 本体線のベクトル (先端から逆方向へ)
                main_vec_end_x, main_vec_end_y = main_p2 if tip_candidate_idx == 0 else main_p1
                
                # 矢印の先端 (tip_x, tip_y) から本体線が伸びる方向の角度
                main_vec_angle = get_line_orientation_angle((tip_x, tip_y), (main_vec_end_x, main_vec_end_y))
                
                arrow_head_lines_with_info = [] # (線分座標, 角度, インデックス) を格納

                # 矢羽根候補を探す
                for j, (hx1, hy1, hx2, hy2) in enumerate(lines_list):
                    if i == j or j in used_line_indices:
                        continue

                    head_len = dist((hx1, hy1), (hx2, hy2))
                    # 矢羽根が本体線に対して長すぎる、または短すぎる場合はスキップ
                    if head_len > main_len * arrow_head_max_len_ratio or head_len < 5: 
                        continue

                    # 矢羽根が先端候補に接続しているかチェック
                    is_connected = False
                    head_other_end = (0,0) # 接続していない側の端点
                    head_start_point = (0,0) # 接続している側の端点

                    if dist((hx1, hy1), (tip_x, tip_y)) < arrow_connection_tolerance:
                        is_connected = True
                        head_start_point = (hx1, hy1)
                        head_other_end = (hx2, hy2)
                    elif dist((hx2, hy2), (tip_x, tip_y)) < arrow_connection_tolerance:
                        is_connected = True
                        head_start_point = (hx2, hy2)
                        head_other_end = (hx1, hy1)
                    
                    if is_connected:
                        # 矢羽根の接続点からの角度
                        head_angle = get_line_orientation_angle(head_start_point, head_other_end)

                        # 本体線から矢羽根への角度差
                        angle_diff = abs(main_vec_angle - head_angle)
                        angle_diff = min(angle_diff, 360 - angle_diff) # 鋭角の方を取る

                        # 矢羽根の開き角度が適切な範囲内か
                        if arrow_head_angle_min <= angle_diff <= arrow_head_angle_max:
                            arrow_head_lines_with_info.append(((hx1, hy1, hx2, hy2), head_angle, j))
                
                # 矢羽根が2本以上見つかったら矢印と判定
                if len(arrow_head_lines_with_info) >= 2:
                    # 最初の2本の矢羽根を使用（より洗練された選択ロジックも考えられる）
                    ah1_coords, angle1_raw, ah1_idx = arrow_head_lines_with_info[0]
                    ah2_coords, angle2_raw, ah2_idx = arrow_head_lines_with_info[1]

                    # 2本の矢羽根がおおよそ対称に開いていることを確認 (より堅牢なチェック)
                    # 本体線の方向 (main_vec_angle) を基準とした相対角度
                    rel_angle1 = (angle1_raw - main_vec_angle + 360) % 360
                    rel_angle2 = (angle2_raw - main_vec_angle + 360) % 360
                    
                    # 2つの相対角度が180度付近の差があるか (本体線を挟んで開いているか)
                    # 許容範囲を広めに取る (例: 100度から260度の間)
                    angle_span = abs(rel_angle1 - rel_angle2)
                    if 100 < angle_span < 260: # この範囲で矢羽根の対称性を確認
                        # 矢印として検出！描画
                        cv2.line(output_image, main_p1, main_p2, (255, 0, 255), 3) # 本体線（マゼンタ）
                        cv2.line(output_image, (ah1_coords[0], ah1_coords[1]), (ah1_coords[2], ah1_coords[3]), (255, 0, 255), 3)
                        cv2.line(output_image, (ah2_coords[0], ah2_coords[1]), (ah2_coords[2], ah2_coords[3]), (255, 0, 255), 3)
                        
                        cv2.putText(output_image, "Arrow", (tip_x, tip_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        detected_elements.append("矢印")
                        used_line_indices.add(i) # 本体線を使用済み
                        used_line_indices.add(ah1_idx) # 矢羽根1を使用済み
                        used_line_indices.add(ah2_idx) # 矢羽根2を使用済み
                        
                        # 矢印が見つかったら、この本体線からの矢印検出は終了し、次の本体線へ
                        break 

        # 矢印として使われなかった線分を直線として描画
        for i, (x1, y1, x2, y2) in enumerate(lines_list):
            if i in used_line_indices:
                continue # 矢印の構成要素として使われた線はスキップ

            line_length = dist((x1,y1), (x2,y2))
            if line_length < min_line_length_hough: # 短すぎる直線は無視
                continue

            angle_deg = get_line_orientation_angle((x1,y1), (x2,y2)) % 180 # 0-180度に正規化

            # 水平・垂直・斜線の分類
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
            # else はデフォルトの斜線

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

# 初期デフォルト値をセッションステートに保存
# これらはdetection_sensitivityで調整される元の値として使われる
if 'base_hough_threshold' not in st.session_state:
    st.session_state.base_hough_threshold = 10 # Hough変換の最小交点数のベース値 (感度で調整)
    st.session_state.base_min_line_length_hough = 10 # 線分の最小長さのベース値 (感度で調整)
    st.session_state.base_min_triangle_area = 500 # 最小三角形面積のベース値 (感度で調整)
    st.session_state.base_approx_epsilon_factor = 0.05 # 輪郭近似の精度のベース値 (感度で調整)
    st.session_state.base_arrow_head_max_len_ratio = 0.4 # 矢羽根の最大長比率のベース値 (感度で調整)
    st.session_state.base_arrow_connection_tolerance = 5 # 矢羽根接続許容誤差のベース値 (感度で調整)
    st.session_state.base_line_angle_tolerance = 5 # 直線判定角度許容誤差のベース値 (感度で調整)


if uploaded_file is not None:
    input_image_pil = Image.open(uploaded_file)
    st.subheader("アップロードされた画像:")
    st.image(input_image_pil, caption="元の画像", use_column_width=True)

    st.sidebar.header("パラメータ調整")

    # --- 全体の検出感度 ---
    st.sidebar.markdown("### 全体の検出感度")
    detection_sensitivity_val = st.sidebar.slider(
        "全体の検出感度", min_value=0.1, max_value=1.0, value=0.5, step=0.05, format="%.2f",
        help="スライダーを右に動かすと、より多くの要素（小さいものや弱い線）を検出しますが、ノイズを拾いやすくなります。左に動かすと、より厳密な検出になり、安定しますが、検出漏れが生じる可能性もあります。これが最も重要な調整スライダーです。"
    )

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
    canny_lower_base = st.sidebar.slider("Canny下限閾値ベース", 0, 255, 50, 1,
        help="Cannyエッジ検出の下限閾値の基準値です。全体の検出感度によって最終的な値が調整されます。手書き線が薄い場合はこの値を下げてみてください。"
    )
    canny_upper_base = st.sidebar.slider("Canny上限閾値ベース", 0, 255, 150, 1,
        help="Cannyエッジ検出の上限閾値の基準値です。全体の検出感度によって最終的な値が調整されます。手書き線が太い場合やエッジが複数検出される場合はこの値を上げてみてください。"
    )

    # --- 4. 線分の補強（モルフォロジー変換） ---
    st.sidebar.markdown("### 4. 線分の補強（クロージング）")
    morph_kernel_size_val = st.sidebar.slider(
        "クロージング カーネルサイズ", min_value=1, max_value=7, value=1, step=2, # 奇数のみ (1は実質オフ)
        help="二値化後の途切れた線を繋げたり、小さな穴を埋めたりします。値を大きくすると効果が強まります。"
    )

    # --- 直線検出の個別調整パラメータ ---
    st.sidebar.markdown("### 直線検出の個別調整")
    max_line_gap_hough_manual_val = st.sidebar.slider(
        "線分間の最大ギャップ (Max Line Gap)", min_value=0, max_value=100, value=20, step=1,
        help="Hough変換で同一直線上と見なされる線分間の最大許容ギャップです。この値を**大きくする**と、途切れた線も一本の長い直線として認識されやすくなります。手書きの線が途切れがちな場合に非常に重要です。"
    )

    # --- 矢印の角度条件のみ手動調整可能にする ---
    st.sidebar.markdown("### 矢印の角度条件（個別に調整）")
    arrow_head_angle_min_val = st.sidebar.slider(
        "矢羽根の最小開き角度 (度)", min_value=10, max_value=60, value=20, step=1,
        help="矢羽根の線が本体線から開く最小角度。手書きの揺らぎに合わせて調整してください。"
    )
    arrow_head_angle_max_val = st.sidebar.slider(
        "矢羽根の最大開き角度 (度)", min_value=30, max_value=90, value=50, step=1,
        help="矢羽根の線が本体線から開く最大角度。手書きの揺らぎに合わせて調整してください。"
    )

    st.write("---")
    st.write("構造物要素を検出中...")

    # 全検出処理を実行
    processed_image_pil, detected_elements_list = detect_structures(
        input_image_pil,
        resize_scale,
        detection_sensitivity_val,
        blur_kernel_size,
        threshold_method,
        block_size_val, C_value_val,
        canny_lower_base,
        canny_upper_base,
        morph_kernel_size_val,
        max_line_gap_hough_manual_val, # 新しい手動調整パラメータを渡す
        arrow_head_angle_min_val,
        arrow_head_angle_max_val
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