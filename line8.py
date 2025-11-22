import streamlit as st
import numpy as np
import cv2
from PIL import Image

def detect_structure_elements(image: Image.Image,
                              # 前処理パラメータ
                              blur_kernel: int,
                              threshold_type: str,
                              block_size: int, C_value: int,
                              # Cannyエッジ検出パラメータ
                              canny_lower: int, canny_upper: int,
                              # モルフォロジー変換パラメータ
                              morph_kernel_size: int,
                              # 図形検出パラメータ
                              min_object_area: int,
                              approx_epsilon_factor: float,
                              hough_threshold: int,
                              min_line_length: int, max_line_gap: int):
    """
    入力画像から三角形、円、斜線を検出し、検出結果を描画した画像を返します。
    様々な前処理と検出パラメータを調整可能です。
    """
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- 1. ノイズ除去（ガウシアンブラー）---
    # カーネルサイズは奇数である必要があります
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred_image = cv2.GaussianBlur(gray_image, (blur_kernel, blur_kernel), 0)

    # --- 2. 二値化 ---
    # 手書きの線は濃度が不均一な場合があるため、適応的閾値処理が有効です。
    # block_sizeは奇数である必要があります
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    
    if threshold_type == "Otsu":
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # OtsuはTHRESH_BINARYを使うので、前景が白、背景が黒になります
        # CannyやfindContoursのために線を白（255）として、背景を黒（0）にする
        binary_image = cv2.bitwise_not(binary_image) # 反転
    elif threshold_type == "Adaptive Mean":
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, block_size, C_value)
    elif threshold_type == "Adaptive Gaussian":
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, block_size, C_value)
    else: # Simple Binary (あまり推奨しないがオプションとして残す)
        _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    # --- 3. Cannyエッジ検出 ---
    # 二値化された画像に対して行うことで、より鮮明なエッジが得られやすい
    edges = cv2.Canny(binary_image, canny_lower, canny_upper)

    # --- 4. モルフォロジー変換（クロージング）---
    # 手書きの途切れやすい線を繋げるのに有効
    # カーネルサイズは奇数である必要があります
    morph_kernel_size = morph_kernel_size if morph_kernel_size % 2 == 1 else morph_kernel_size + 1
    if morph_kernel_size > 1: # 1の場合は実質何もしない
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        # クロージング = 膨張 -> 収縮: 途切れた線を繋ぎ、小さな穴を埋める
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    output_image = np.copy(image_np) # 結果描画用の画像コピー（RGB）
    detected_elements = []

    # --- 5. 輪郭検出に基づく図形（三角形、円）の検出 ---
    # `cv2.RETR_EXTERNAL` は最も外側の輪郭のみ、`cv2.CHAIN_APPROX_SIMPLE` は簡略化された輪郭
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 最小面積でノイズや小さすぎる要素を除去
        if area < min_object_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        
        # 輪郭を多角形に近似
        epsilon = approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        num_vertices = len(approx)
        shape_name = None # 検出された図形名
        color = (255, 255, 255) # デフォルトの色（白）

        # 図形の分類
        if num_vertices == 3:
            shape_name = "三角形"
            color = (0, 255, 0) # 緑
            
        elif num_vertices >= 8: # 頂点が多い場合は円の可能性
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # 円形度（面積比）とサイズのチェック
            circle_area = np.pi * radius**2
            # 輪郭の面積が外接円の面積の80%以上かつ、ある程度の大きさがある場合
            if circle_area > 0 and area / circle_area > 0.80 and radius > 5: # 半径5ピクセル未満は無視
                shape_name = "円"
                color = (0, 0, 255) # 赤
                cv2.circle(output_image, center, radius, color, 4) # 円を描画（approxではなくminEnclosingCircleの円）
            
        if shape_name: # 三角形または円が検出された場合
            cv2.drawContours(output_image, [approx], 0, color, 4) # 輪郭を描画
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_image, shape_name, (cX - 30, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                detected_elements.append(shape_name)

    # --- 6. 斜線の検出（Hough変換による線分検出の利用）---
    # エッジ画像から直接線分を検出
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 線の傾きをチェックして「斜線」を特定
            # ほぼ水平（±5度）またはほぼ垂直（90±5度）ではない線を斜線とみなす
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            # abs(angle_deg % 180) は 0-180 の範囲に正規化された角度（0度と180度は水平）
            # abs(angle_deg % 90) もしくは abs(angle_deg - 90) のチェックは垂直
            
            # 水平または垂直と判断する閾値（例: 5度）
            angle_tolerance = 5 
            
            is_horizontal = (abs(angle_deg) < angle_tolerance) or (abs(angle_deg - 180) < angle_tolerance) or (abs(angle_deg + 180) < angle_tolerance)
            is_vertical = (abs(angle_deg - 90) < angle_tolerance) or (abs(angle_deg + 90) < angle_tolerance)
            
            if not (is_horizontal or is_vertical):
                cv2.line(output_image, (x1, y1), (x2, y2), (255, 165, 0), 2) # オレンジ色で斜線を描画
                detected_elements.append(f"斜線") # 検出リストにはシンプルに「斜線」と追加

    # 結果画像をPIL形式に戻して返す
    return Image.fromarray(output_image), detected_elements

# --- Streamlit UIの構築 ---
st.set_page_config(layout="wide", page_title="構造物要素検出アプリ")

st.title("手書き対応：構造物要素検出アプリ")
st.write("画像をアップロードし、左のサイドバーでパラメータを調整して、**三角形、円、斜線**の検出を試してください。")
st.write("特に**手書き画像**の場合、[1. ノイズ除去]、[2. 二値化]、[4. 線分の補強] のパラメータ調整が重要です。")

uploaded_file = st.file_uploader("ここに画像をドラッグ＆ドロップするか、ファイルを選択してください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image_pil = Image.open(uploaded_file)
    st.subheader("アップロードされた画像:")
    st.image(input_image_pil, caption="元の画像", use_column_width=True)

    st.sidebar.header("前処理と検出パラメータ")

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
    
    # 適応的閾値用のパラメータ（Otsu/Simple Binary 選択時は非表示）
    if threshold_method in ["Adaptive Mean", "Adaptive Gaussian"]:
        block_size = st.sidebar.slider(
            "適応的閾値のブロックサイズ", min_value=3, max_value=51, value=11, step=2, # 奇数のみ
            help="周囲の平均を計算する領域のサイズです。線の太さや背景のムラに応じて調整。"
        )
        C_value = st.sidebar.slider(
            "適応的閾値のC値", min_value=-10, max_value=10, value=2, step=1,
            help="平均値から引かれる定数。明るい背景に暗い線の場合、正の値が有効です。"
        )
    else:
        # 非表示だが、内部でデフォルト値を渡す
        block_size = 11
        C_value = 2

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
    morph_kernel_size = st.sidebar.slider(
        "クロージング カーネルサイズ", min_value=1, max_value=7, value=1, step=2, # 奇数のみ (1は実質オフ)
        help="二値化後の途切れた線を繋げたり、小さな穴を埋めたりします。値を大きくすると効果が強まります。"
    )

    # --- 5. 図形検出の最終調整 ---
    st.sidebar.markdown("### 5. 図形検出の最終調整")
    min_object_area = st.sidebar.slider(
        "最小オブジェクト面積 (px^2)", min_value=10, max_value=5000, value=500, step=10,
        help="これより小さい面積のオブジェクト（三角形、円など）は無視されます。ノイズ除去に役立ちます。"
    )
    approx_epsilon = st.sidebar.slider(
        "輪郭近似の精度 (0.01 - 0.20)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f",
        help="値が小さいほど元の輪郭に忠実になりますが、手書きの揺らぎを拾いすぎることがあります。値を大きくするとより単純な形状に近似され、手書きの不正確さを吸収できます。"
    )

    # --- 6. 斜線検出 (Hough Lines P) ---
    st.sidebar.markdown("### 6. 斜線検出 (Hough Lines P)")
    hough_threshold = st.sidebar.slider(
        "Hough変換の最小交点数", min_value=1, max_value=100, value=10, step=1,
        help="この値が大きいほど、よりはっきりした直線のみが検出されます。"
    )
    min_line_length = st.sidebar.slider(
        "線分の最小長さ", min_value=1, max_value=200, value=10, step=1,
        help="これより短い線分はノイズと見なされ、検出されません。"
    )
    max_line_gap = st.sidebar.slider(
        "線分間の最大ギャップ", min_value=0, max_value=100, value=8, step=1,
        help="この値の範囲内で途切れている線分は、1本の直線として扱われます。"
    )

    st.write("---")
    st.write("構造物要素を検出中...")

    # 全検出処理を実行
    processed_image_pil, detected_elements_list = detect_structure_elements(
        input_image_pil,
        blur_kernel_size,
        threshold_method,
        block_size, C_value,
        canny_lower_threshold,
        canny_upper_threshold,
        morph_kernel_size,
        min_object_area,
        approx_epsilon,
        hough_threshold,
        min_line_length,
        max_line_gap
    )

    st.subheader("検出結果:")
    st.image(processed_image_pil, caption="検出された構造物要素", use_column_width=True)

    # 検出された要素のリスト表示
    if detected_elements_list:
        st.subheader("検出された要素の概要:")
        # 要素の種類ごとのカウント
        from collections import Counter
        element_counts = Counter(detected_elements_list)
        for element, count in element_counts.items():
            st.write(f"- {element}: {count}個")
    else:
        st.write("指定された図形要素は検出されませんでした。サイドバーのパラメータを調整してみてください。")

st.markdown("---")
st.write("© 2025 構造物要素検出アプリ")