import streamlit as st
import numpy as np
import cv2
from PIL import Image

def detect_and_draw_lines(image: Image.Image, hough_threshold: int, min_line_length: int, max_line_gap: int, use_canny: bool = False):
    """
    画像をHough変換で処理し、線分を検出して描画した画像を返します。

    Args:
        image (PIL.Image.Image): 入力画像（PIL形式）。
        hough_threshold (int): Hough変換の閾値。
        min_line_length (int): 検出する線分の最小長さ。
        max_line_gap (int): 同一直線上と見なす線分間の最大ギャップ。
        use_canny (bool): Cannyエッジ検出を使用するかどうか。Falseの場合はSobelを使用。

    Returns:
        tuple: (線分が描画されたPIL画像, 検出された線分の座標リスト)。
    """
    # PIL画像をNumPy配列（RGB）に変換し、OpenCVのBGR形式に変換
    image_np = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # エッジ検出の選択
    if use_canny:
        # Cannyエッジ検出は一般的なエッジ抽出
        edges = cv2.Canny(gray_image, 50, 150)
    else:
        # Sobelフィルタはグラデーション情報をより保持し、Hough変換と相性が良い場合があります
        sobel_x = cv2.Sobel(gray_image, cv2.CV_8U, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_8U, 0, 1, ksize=3)
        edges = cv2.bitwise_or(sobel_x, sobel_y) # X方向とY方向のエッジを結合

    # 確率的Hough変換（HoughLinesP）で線分を検出
    # rho: 距離分解能 (ピクセル単位)
    # theta: 角度分解能 (ラジアン単位)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    output_image = np.copy(image_np) # 結果描画用の画像コピー（RGB）
    detected_lines_coords = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] # 線分の端点座標
            cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2) # 赤色の線（太さ2）で描画
            detected_lines_coords.append(((x1, y1), (x2, y2)))

    # 結果画像をPIL形式に戻して返す
    return Image.fromarray(output_image), detected_lines_coords

# --- Streamlit UIの構築 ---
st.set_page_config(layout="centered", page_title="線分検出アプリ")

st.title("画像の線分検出アプリ")
st.write("画像をアップロードし、左のサイドバーでパラメータを調整して線分検出を試してください。")

# 画像アップロードエリア
uploaded_file = st.file_uploader("ここに画像をドラッグ＆ドロップするか、ファイルを選択してください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.subheader("アップロードされた画像:")
    st.image(input_image, caption="元の画像", use_column_width=True)

    st.sidebar.header("検出パラメータ")

    # エッジ検出方法の選択
    use_canny_edge = st.sidebar.checkbox("Cannyエッジ検出を使用", value=False, help="チェックすると、より一般的なCannyエッジ検出を使用します。チェックを外すと、Sobelフィルタでエッジを検出します。")

    # Hough変換のパラメータスライダー
    hough_threshold = st.sidebar.slider(
        "Hough変換の最小交点数 (Threshold)",
        min_value=10, max_value=200, value=50, step=1,
        help="この値が大きいほど、よりはっきりした直線のみが検出されます。"
    )
    min_line_length = st.sidebar.slider(
        "線分の最小長さ (Min Line Length)",
        min_value=10, max_value=300, value=50, step=1,
        help="これより短い線分はノイズと見なされ、検出されません。"
    )
    max_line_gap = st.sidebar.slider(
        "線分間の最大ギャップ (Max Line Gap)",
        min_value=0, max_value=100, value=10, step=1,
        help="この値の範囲内で途切れている線分は、1本の直線として扱われます。"
    )

    st.write("---")
    st.write("線分を検出中...")

    # 線分検出を実行
    processed_image_pil, detected_lines = detect_and_draw_lines(
        input_image,
        hough_threshold,
        min_line_length,
        max_line_gap,
        use_canny_edge
    )

    st.subheader("検出結果:")
    st.image(processed_image_pil, caption="検出された線分", use_column_width=True)

    # 検出された線分の座標を表示
    if detected_lines:
        st.subheader("検出された線分の座標:")
        # 最初から数本だけ表示して、多すぎる場合に備える
        for i, (start, end) in enumerate(detected_lines[:10]):
            st.write(f"線分 {i+1}: 始点 {start}, 終点 {end}")
        if len(detected_lines) > 10:
            st.write(f"...他にも{len(detected_lines) - 10}本の線分が検出されました。")
    else:
        st.write("画像から直線は検出されませんでした。サイドバーのパラメータを調整してみてください。")

st.markdown("---")
st.write("© 2025 線分検出アプリ")