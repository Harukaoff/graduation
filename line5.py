import streamlit as st
import numpy as np
import cv2
from PIL import Image

def detect_specific_shapes(image: Image.Image, hough_threshold: int, min_line_length: int, max_line_gap: int, canny_lower: int, canny_upper: int):
    """
    画像を処理し、三角形、円、斜線を検出して描画した画像を返します。
    """
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- エッジ検出（Cannyを使用）---
    # 図形検出にはCannyエッジが一般的に適しています
    edges = cv2.Canny(gray_image, canny_lower, canny_upper)

    output_image = np.copy(image_np) # 結果描画用の画像コピー（RGB）
    detected_elements = []

    # --- 1. 斜線の検出（Hough変換による線分検出の利用）---
    # 前回成功したパラメータを初期値として使用
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
            # 線の傾きをチェックして「斜線」を特定する例
            # 完全に水平・垂直ではない線を斜線とみなす
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle_rad)
            # ほぼ水平または垂直の線を除外（閾値は調整が必要）
            if not ((abs(angle_deg) < 5) or (abs(angle_deg - 90) < 5) or (abs(angle_deg + 90) < 5) or (abs(angle_deg - 180) < 5) or (abs(angle_deg + 180) < 5)):
                cv2.line(output_image, (x1, y1), (x2, y2), (255, 165, 0), 2) # オレンジ色で斜線を描画
                detected_elements.append(f"斜線: ({x1},{y1}) - ({x2},{y2})")

    # --- 2. 輪郭検出に基づく図形（三角形、円）の検出 ---
    # `cv2.RETR_EXTERNAL` は外側の輪郭のみ、`cv2.CHAIN_APPROX_SIMPLE` は簡略化された輪郭
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300: # 小さすぎる輪郭は無視（閾値調整）
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.04 * perimeter # 輪郭近似の精度（調整可能）
        approx = cv2.approxPolyDP(contour, epsilon, True) # 輪郭を多角形に近似

        num_vertices = len(approx)
        shape_name = "Unknown"
        color = (255, 255, 255) # デフォルトの色（白）

        # 図形の分類
        if num_vertices == 3:
            shape_name = "三角形"
            color = (0, 255, 0) # 緑
            cv2.drawContours(output_image, [approx], 0, color, 4)
            
        elif num_vertices >= 8: # 頂点が多い場合は円の可能性
            # 円の検出を試みる (HoughCirclesも検討の価値あり)
            # ここでは外接円の計算と、面積による「円らしさ」の確認
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # 円形度（面積比）で精度を上げる
            circle_area = np.pi * radius**2
            # 輪郭の面積が外接円の面積の80%以上かつ、ある程度の大きさがある場合
            if area / circle_area > 0.8 and radius > 15: # 半径15ピクセル未満の小さい円は無視
                shape_name = "円"
                color = (0, 0, 255) # 赤
                cv2.circle(output_image, center, radius, color, 4)

        if shape_name != "Unknown":
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_image, shape_name, (cX - 30, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                detected_elements.append(shape_name)

    return Image.fromarray(output_image), detected_elements

# --- Streamlit UIの構築 ---
st.set_page_config(layout="centered", page_title="構造物要素検出アプリ")

st.title("構造物要素検出アプリ")
st.write("画像をアップロードし、左のサイドバーでパラメータを調整して、**三角形、円、斜線**の検出を試してください。")

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

    # Hough変換のパラメータ（斜線検出用）
    st.sidebar.markdown("### 斜線検出 (Hough Lines P)")
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
    st.write("図形要素を検出中...")

    # 図形検出を実行
    processed_image_pil, detected_elements_list = detect_specific_shapes(
        input_image_pil,
        hough_threshold,
        min_line_length,
        max_line_gap,
        canny_lower_threshold,
        canny_upper_threshold
    )

    st.subheader("検出結果:")
    st.image(processed_image_pil, caption="検出された図形要素", use_column_width=True)

    # 検出された要素のリスト表示
    if detected_elements_list:
        st.subheader("検出された要素:")
        for element in sorted(list(set(detected_elements_list))): # 重複を削除してソート
            st.write(f"- {element}")
    else:
        st.write("指定された図形要素は検出されませんでした。サイドバーのパラメータを調整してみてください。")

st.markdown("---")
st.write("© 2025 構造物要素検出アプリ")