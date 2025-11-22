import streamlit as st
import numpy as np
import cv2
from PIL import Image

def detect_triangles_handdrawn(image: Image.Image,
                               blur_kernel: int,
                               threshold_type: str,
                               block_size: int, C_value: int,
                               canny_lower: int, canny_upper: int,
                               min_area: int, approx_epsilon_factor: float):
    """
    手書き画像から三角形を検出し、検出結果を描画した画像を返します。
    前処理（ノイズ除去、二値化）とCannyエッジ検出のパラメータを調整可能です。
    """
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- 1. ノイズ除去 ---
    # ガウシアンブラーは画像のノイズを滑らかにし、エッジ検出を安定させます。
    # 奇数である必要があるので、偶数の場合は+1
    blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred_image = cv2.GaussianBlur(gray_image, (blur_kernel, blur_kernel), 0)

    # --- 2. 二値化（コントラスト強調）---
    # 手書きの線は濃度が不均一な場合があるため、適応的閾値処理が有効です。
    # 背景と線を明確に分けます。
    if threshold_type == "Otsu":
        # 大津の二値化: 全体的な最適な閾値を自動で決定
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif threshold_type == "Adaptive Mean":
        # 適応的平均閾値処理: 周囲のピクセル値に基づいて閾値を決定
        # block_sizeは奇数である必要があります
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV, block_size, C_value)
    elif threshold_type == "Adaptive Gaussian":
        # 適応的ガウシアン閾値処理: ガウシアン加重平均に基づいて閾値を決定
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, block_size, C_value)
    else: # "Simple Binary"
        # シンプルな固定閾値。手書きには不向きな場合が多い
        _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Cannyエッジ検出は二値化された画像に対して行うと、より鮮明なエッジが得られやすい
    edges = cv2.Canny(binary_image, canny_lower, canny_upper)

    output_image = np.copy(image_np) # 結果描画用の画像コピー（RGB）
    detected_triangle_count = 0

    # --- 輪郭検出 ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 最小面積でノイズを除去
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        
        # 輪郭を多角形に近似
        # 手書きの場合、approx_epsilon_factorを調整することが重要
        epsilon = approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        num_vertices = len(approx)

        # --- 三角形の判定 ---
        if num_vertices == 3:
            # 輪郭の形状をもう少し厳密にチェックすることも可能
            # 例: cv2.isContourConvex(approx) で凸包かどうか確認
            
            # 三角形を描画（緑色）
            cv2.drawContours(output_image, [approx], 0, (0, 255, 0), 4)
            detected_triangle_count += 1
            
            # 中心座標にテキストを表示
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_image, "Triangle", (cX - 30, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return Image.fromarray(output_image), detected_triangle_count

# --- Streamlit UIの構築 ---
st.set_page_config(layout="wide", page_title="手書き図形（三角形）検出アプリ")

st.title("手書き図形（三角形）検出アプリ")
st.write("画像をアップロードし、左のサイドバーでパラメータを調整して**手書きの三角形**の検出を試してください。")

uploaded_file = st.file_uploader("ここに画像をドラッグ＆ドロップするか、ファイルを選択してください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image_pil = Image.open(uploaded_file)
    st.subheader("アップロードされた画像:")
    st.image(input_image_pil, caption="元の画像", use_column_width=True)

    st.sidebar.header("前処理と検出パラメータ")

    # --- ノイズ除去 ---
    st.sidebar.markdown("### 1. ノイズ除去（ぼかし）")
    blur_kernel_size = st.sidebar.slider(
        "ガウシアンブラー カーネルサイズ", min_value=1, max_value=21, value=5, step=2, # 奇数のみ
        help="値を大きくすると、より強くぼかされノイズが除去されます。線の太さも変わります。"
    )

    # --- 二値化 ---
    st.sidebar.markdown("### 2. 二値化（線をはっきりさせる）")
    threshold_method = st.sidebar.selectbox(
        "二値化方法",
        ("Adaptive Gaussian", "Adaptive Mean", "Otsu", "Simple Binary"),
        help="手書き画像には 'Adaptive Gaussian' や 'Adaptive Mean' がおすすめです。"
    )
    
    # 適応的閾値用のパラメータ
    block_size = st.sidebar.slider(
        "適応的閾値のブロックサイズ", min_value=3, max_value=51, value=11, step=2, # 奇数のみ
        help="周囲の平均を計算する領域のサイズです。線の太さや背景のムラに応じて調整。"
    )
    C_value = st.sidebar.slider(
        "適応的閾値のC値", min_value=-10, max_value=10, value=2, step=1,
        help="平均値から引かれる定数。明るい背景に暗い線の場合、正の値が有効です。"
    )

    # --- Cannyエッジ検出 ---
    st.sidebar.markdown("### 3. Cannyエッジ検出")
    canny_lower_threshold = st.sidebar.slider("Canny下限閾値", 0, 255, 50, 1,
        help="低い値ほど細かいエッジも検出します。ノイズが多くなる可能性も。"
    )
    canny_upper_threshold = st.sidebar.slider("Canny上限閾値", 0, 255, 150, 1,
        help="高い値ほど強いエッジのみ検出します。線の途切れが発生する可能性も。"
    )

    # --- 三角形検出の最終調整 ---
    st.sidebar.markdown("### 4. 三角形検出の調整")
    min_triangle_area = st.sidebar.slider(
        "最小三角形面積 (px^2)", min_value=10, max_value=5000, value=500, step=10,
        help="これより小さい面積の三角形は無視されます。手書きの場合、この値は調整が必要です。"
    )
    approx_epsilon = st.sidebar.slider(
        "輪郭近似の精度 (0.01 - 0.20)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, format="%.2f",
        help="値が小さいほど元の輪郭に忠実になりますが、手書きの揺らぎを拾いすぎることがあります。値を大きくするとより単純な形状に近似され、手書きの不正確さを吸収できます。"
    )

    st.write("---")
    st.write("三角形を検出中...")

    # 三角形検出を実行
    processed_image_pil, num_triangles = detect_triangles_handdrawn(
        input_image_pil,
        blur_kernel_size,
        threshold_method,
        block_size, C_value,
        canny_lower_threshold,
        canny_upper_threshold,
        min_triangle_area,
        approx_epsilon
    )

    st.subheader("検出結果:")
    st.image(processed_image_pil, caption="検出された三角形", use_column_width=True)

    st.subheader(f"検出された三角形の数: {num_triangles}個")
    if num_triangles == 0:
        st.write("三角形は検出されませんでした。サイドバーのパラメータを調整してみてください。")

st.markdown("---")
st.write("© 2025 手書き図形検出アプリ")