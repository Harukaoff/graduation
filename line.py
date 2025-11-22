import streamlit as st
import cv2
import numpy as np
import math
import pandas as pd # pandasが他の場所で使われている場合を想定

# --- 輪郭検出関数 ---
def detect_contours(gray_img, min_area, max_area):
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # 二値化 (記号が黒、背景が白の場合のために反転)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 面積で輪郭をフィルタリング
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    return filtered_contours, thr

# --- PCAで主軸を抽出し描画する関数 ---
def draw_pca(img, cnt, color=(255, 0, 0)):
    if len(cnt) >= 5: # PCAには最低5点が必要
        moments = cv2.moments(cnt)
        if moments['m00'] == 0: # ゼロ除算を避ける
            return

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # PCAの計算
        cnt_f32 = np.float32(cnt.reshape(-1, 2))
        
        # PCA計算が可能な最小点数を確認
        if cnt_f32.shape[0] < 2:
            return

        try:
            _, eigenvectors = cv2.PCACompute(cnt_f32, mean=None, maxComponents=1)
        except cv2.error:
            # PCA計算中にエラーが発生した場合
            return

        if eigenvectors.shape[0] < 1: 
            return

        # 主軸の角度を計算
        angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0])

        # 線の端点を定義 (長さを動的に調整)
        line_length = max(cnt.shape[0] / 5, 20) 
        end_x = int(cx + line_length * math.cos(angle))
        end_y = int(cy + line_length * math.sin(angle))
        start_x = int(cx - line_length * math.cos(angle))
        start_y = int(cy - line_length * math.sin(angle))

        cv2.line(img, (start_x, start_y), (end_x, end_y), color, 2)


# --- 幾何学的特徴に基づく記号分類関数 ---
def classify_symbol_by_geometry(contour, image_width, image_height):
    # 外接矩形を取得
    x, y, w, h = cv2.boundingRect(contour)
    
    # 面積
    area = cv2.contourArea(contour)
    # 外周長
    perimeter = cv2.arcLength(contour, True)
    
    # 小さすぎる、または大きすぎる輪郭は無視
    # 画像全体の1/10より大きいものは通常記号ではない
    if area < 50 or area > image_width * image_height / 10:
        return "Unknown", 0.0

    # 形状近似 (ポリゴン近似)
    # 精度パラメータ epsilon は、輪郭の周長に応じて調整
    epsilon = 0.04 * perimeter 
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    label = "Unknown"
    confidence = 0.0 # 幾何学的特徴からの信頼度は0-1で表現しにくいが、便宜上

    # --- 各要素の幾何学的特徴に基づく判別 ---

    # 1. ピン支点 (三角形)
    # 3つの頂点を持つ多角形 (三角形) を検出
    if num_vertices == 3:
        # さらに、三角形の底辺がほぼ水平であることを確認するなどの工夫も可能
        label = "pin_candidate"
        confidence = 0.8 # 暫定的な信頼度

    # 2. ヒンジ (円)
    # 円に近い形状 (外周長と面積の関係、アスペクト比など)
    elif perimeter > 0 and (4 * np.pi * area) / (perimeter * perimeter) > 0.7: # 円形度 (1に近いほど円)
        aspect_ratio = float(w) / h
        if 0.8 < aspect_ratio < 1.2: # アスペクト比が1に近い
            label = "hinge_candidate"
            confidence = 0.7

    # 3. 矢印 (荷重) - 複雑だが、単純なV字や鋭角な形状
    # 頂点数が多い、または特定の角度を持つ頂点群。非常に困難な判別。
    # 単純な多角形近似では難しいが、ここでは簡易的に「細長く、かつ、凹みがある」形状を試みる。
    elif num_vertices > 4 and (h / w > 1.5 or w / h > 1.5 or h / w < 0.7 or w / h < 0.7): # ある程度細長い形状
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        # 凸包との面積差が大きい場合（凹みがある形状、矢印など）
        if hull_area > 0 and (hull_area - area) / hull_area > 0.3: 
             label = "load_arrow_candidate"
             confidence = 0.6


    # 4. 固定支点 (直線と斜線の組み合わせ)
    # 5. ピンローラー支点 (三角形と隙間と直線)
    # これらは複数の独立した輪郭や、複雑な内部構造を持つため、
    # 単一の輪郭の幾何学的特徴だけでは判別が非常に困難です。
    # より高度な画像処理 (多輪郭の関係性解析、パターン認識モデルなど) が必要となります。
    # 現時点では 'Unknown' となります。

    # 小さすぎる領域はノイズの可能性もあるため、再度確認
    if area < 100: # この閾値は画像サイズや手書きの太さに応じて要調整
        return "Unknown", 0.0

    return label, confidence


# --- Streamlit アプリのUI ---
st.title("手書き構造物認識アプリ (テンプレートなしバージョン)")

uploaded_file = st.file_uploader("手書きの構造物画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # ファイルをNumpy配列として読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    src_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if src_img is None:
        st.error("画像を読み込めませんでした。")
    else:
        st.image(src_img, caption="アップロードされた画像", use_column_width=True)

        gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        output_img = src_img.copy() # 結果を描画する画像

        st.sidebar.header("認識設定")
        min_area = st.sidebar.slider("輪郭の最小面積", 10, 1000, 50)
        max_area = st.sidebar.slider("輪郭の最大面積", 100, 50000, 10000)
        
        # --- 画像の二値化 ---
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        _, binarized_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        st.image(binarized_img, caption="二値化画像", use_column_width=True)

        # --- 線の検出 (Hough Line Transform) ---
        edges = cv2.Canny(binarized_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        
        detected_lines_info = []
        if lines is not None:
            st.subheader("検出された線分 (部材候補)")
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                cv2.line(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2) # 青色で線を描画
                detected_lines_info.append(f"- 線分 {i+1}: ({x1}, {y1}) から ({x2}, {y2})")
            st.markdown("\n".join(detected_lines_info)) # Markdown形式でリスト表示
            st.write(f"検出された線分の数: {len(lines)}")
        else:
            st.write("線分は検出されませんでした。")
        
        # --- 記号（支持条件、荷重など）の検出と幾何学的分類 ---
        cnts, _ = detect_contours(gray_img, min_area, max_area)
        st.write(f"検出された輪郭数 (フィルタリング後): {len(cnts)}")

        detected_elements_info = []
        
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 分類関数に渡す前に、ある程度のフィルタリングを行う
            if w < 10 or h < 10: # 小さすぎる輪郭は無視
                continue

            # 幾何学的特徴に基づいて記号を分類
            label, confidence = classify_symbol_by_geometry(cnt, src_img.shape[1], src_img.shape[0])

            if label != "Unknown" and confidence > 0.5: # 信頼度閾値もここで設定 (例: 0.5)
                # 検出された記号を描画
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2) # 緑色で成功
                text = f"{label} (Conf: {confidence:.2f})"
                cv2.putText(output_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                draw_pca(output_img, cnt) # PCA軸は青
                detected_elements_info.append(f"- **{label}**: 信頼度 {confidence:.2f}, 位置 (x={x}, y={y})")
            else:
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 1) # 赤色で未分類
                cv2.putText(output_img, "?", (x+2, y+h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        st.subheader("検出された記号（幾何学的分類）")
        if detected_elements_info:
            st.markdown("\n".join(detected_elements_info))
        else:
            st.write("記号は検出されませんでした。")
        
        st.markdown("""
        **幾何学的分類の注意点:**
        - **ピン支点 (三角形)**, **ヒンジ (円)** は比較的判別しやすいですが, 手書きのばらつきに左右されます。
        - **荷重 (矢印)** は複雑な形状のため, 簡単な幾何学的特徴だけでは誤認識が多い可能性があります。
        - **ピンローラー支点** や **固定支点** は, 複数の図形要素の組み合わせで構成されるため,
          単一の輪郭の幾何学的特徴のみでは正確な判別は**非常に困難**です。
          これらの判別には, 複数の輪郭間の空間的関係を分析する**より高度なアルゴリズム**や,
          パターン認識のための**機械学習モデルの訓練**が必要になります。
        """)

        st.subheader("最終結果画像")
        st.image(output_img, caption="検出結果", use_column_width=True)