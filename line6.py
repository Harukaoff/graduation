import streamlit as st
import numpy as np
import cv2
from PIL import Image

def detect_triangles(image: Image.Image, canny_lower: int, canny_upper: int, min_area: int, approx_epsilon_factor: float):
    """
    入力画像から三角形を検出し、検出結果を描画した画像を返します。

    Args:
        image (PIL.Image.Image): 入力画像（PIL形式）。
        canny_lower (int): Cannyエッジ検出の下限閾値。
        canny_upper (int): Cannyエッジ検出の上限閾値。
        min_area (int): 検出する三角形の最小面積。これより小さい三角形は無視されます。
        approx_epsilon_factor (float): 輪郭近似の精度を決定する係数。

    Returns:
        tuple: (三角形が描画されたPIL画像, 検出された三角形の数)。
    """
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- Cannyエッジ検出 ---
    # 三角形の輪郭を捉えるために非常に重要
    edges = cv2.Canny(gray_image, canny_lower, canny_upper)

    output_image = np.copy(image_np) # 結果描画用の画像コピー（RGB）
    detected_triangle_count = 0

    # --- 輪郭検出 ---
    # cv2.RETR_EXTERNAL: 最も外側の輪郭のみを検出
    # cv2.CHAIN_APPROX_SIMPLE: 簡略化された輪郭 (メモリ節約)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 最小面積でノイズを除去
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        
        # 輪郭を多角形に近似
        # approx_epsilon_factor を使って輪郭の周囲長に対する比率でepsilonを設定
        epsilon = approx_epsilon_factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        num_vertices = len(approx)

        # --- 三角形の判定 ---
        # 頂点数が3であること、かつ輪郭が閉じていること (True)
        if num_vertices == 3:
            # 輪郭の面積と近似後の面積を比較する追加チェックも可能だが、
            # まずはシンプルな頂点数と最小面積で確認
            
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

    # 結果画像をPIL形式に戻して返す
    return Image.fromarray(output_image), detected_triangle_count

# --- Streamlit UIの構築 ---
st.set_page_config(layout="centered", page_title="三角形検出アプリ")

st.title("三角形検出アプリ")
st.write("画像をアップロードし、左のサイドバーでパラメータを調整して**三角形**の検出を試してください。")

uploaded_file = st.file_uploader("ここに画像をドラッグ＆ドロップするか、ファイルを選択してください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image_pil = Image.open(uploaded_file)
    st.subheader("アップロードされた画像:")
    st.image(input_image_pil, caption="元の画像", use_column_width=True)

    st.sidebar.header("検出パラメータ")

    # Cannyエッジ検出のパラメータ
    st.sidebar.markdown("### Cannyエッジ検出")
    canny_lower_threshold = st.sidebar.slider("Canny下限閾値", 0, 255, 100, 1)
    canny_upper_threshold = st.sidebar.slider("Canny上限閾値", 0, 255, 200, 1)

    # 三角形検出のパラメータ
    st.sidebar.markdown("### 三角形検出の調整")
    min_triangle_area = st.sidebar.slider(
        "最小三角形面積 (px^2)", min_value=10, max_value=2000, value=300, step=10,
        help="これより小さい面積の三角形は無視されます。ノイズ除去に役立ちます。"
    )
    approx_epsilon = st.sidebar.slider(
        "輪郭近似の精度 (0.01 - 0.1)", min_value=0.01, max_value=0.1, value=0.04, step=0.01, format="%.2f",
        help="値が小さいほど、より忠実に輪郭を近似します。大きいと単純な形状になります。"
    )

    st.write("---")
    st.write("三角形を検出中...")

    # 三角形検出を実行
    processed_image_pil, num_triangles = detect_triangles(
        input_image_pil,
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
st.write("© 2025 三角形検出アプリ")