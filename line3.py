import streamlit as st
import cv2
import numpy as np
import math
import pandas as pd # pandasが他の場所で使用されている場合のために残します

# --- 輪郭検出関数 ---
def detect_contours(gray_img, min_area, max_area):
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # 通常、手書きの図面は黒い線で書かれているため、それを前景（白）にするにはTHRESH_BINARY_INVを使用
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
        cy = int(moments['m01'] / moments['m00']) # 修正: m01 / m00 に変更

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


# --- 幾何学的特徴に基づく要素分類関数 (線分と記号を統一して判別) ---
def classify_element_by_geometry(contour, image_width, image_height, min_line_length_px):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # 非常に小さい、または非常に大きい輪郭はノイズまたは背景の一部と見なす
    if area < 50 or area > image_width * image_height / 5: # 画像全体の1/5より大きいものは通常記号ではない
        return "Unknown", 0.0

    aspect_ratio = float(w) / h if h > 0 else 0
    
    # 形状近似 (ポリゴン近似)
    # 線分の検出精度を上げるため、epsilonを小さくする
    epsilon = 0.01 * perimeter 
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    label = "Unknown"
    confidence = 0.0

    # --- 1. 構造部材 (線分) の判別 ---
    # 長くて細い、直線に近い形状
    # minAreaRectを使用して回転を考慮した幅と高さを取得
    rect = cv2.minAreaRect(contour)
    # rect[1] は (width, height)
    width_rect = min(rect[1]) # 短い方の辺
    height_rect = max(rect[1]) # 長い方の辺
    
    if width_rect > 0:
        ratio_rect = height_rect / width_rect # 長い辺 / 短い辺
    else:
        ratio_rect = 0

    # 線分と判断する基準: 非常に高いアスペクト比、短い幅、ほぼ長方形 (num_verticesも考慮)
    # min_line_length_px より長い線分であることも条件に加える
    if height_rect > min_line_length_px and width_rect < 15 and ratio_rect > 8 and num_vertices <= 4:
        # さらに、近似されたポリゴンがほぼ直線（2頂点）であるか確認
        # ただし、手書き線はギザギザになりやすく3-4頂点になることもある
        label = "Member_Line"
        confidence = 0.9
        return label, confidence # 線分を優先的に判別

    # --- 2. ヒンジ (円) の判別 ---
    (cx_c, cy_c), radius = cv2.minEnclosingCircle(contour)
    if radius > 0 and area > 0: # ゼロ除算を避ける
        circularity = area / (math.pi * (radius**2))
        # 円形度とアスペクト比で円を判別。範囲を調整して柔軟性を持たせる
        if 0.75 < circularity < 1.25 and 0.7 < aspect_ratio < 1.3: 
            label = "Hinge_Candidate"
            confidence = 0.8
            return label, confidence

    # --- 3. ピン支点 (三角形) の判別 ---
    # 頂点数が3、かつ凸包であることなどを利用
    if num_vertices == 3 and cv2.isContourConvex(contour): # 凸状の三角形であることを確認
        label = "Pin_Candidate"
        confidence = 0.7
        return label, confidence

    # --- 4. 矢印 (荷重) の判別 ---
    # 非常に難しい判別。単純な幾何学的特徴では誤認識が多い。
    # 例: ある程度細長く (アスペクト比1.5〜5程度)、かつ、凸包でない（凹みがある）形状を試みる
    if num_vertices > 4 and 1.5 < aspect_ratio < 5.0:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0 and (hull_area - area) / hull_area > 0.2: # 凸包との面積差が大きい場合、凹みがある可能性 (矢印のくぼみなど)
            label = "Load_Arrow_Candidate"
            confidence = 0.6
            return label, confidence

    # --- 5. ピンローラー支点 / 固定支点 の判別 ---
    # これらの記号は複数の形状（線と三角形、線と斜線など）の組み合わせであるため、
    # 単一の輪郭から幾何学的特徴のみで判別するのは非常に困難です。
    # これらの検出には、複数の小さな輪郭の相対位置関係を分析したり、
    # 複雑なパターン認識アルゴリズム（機械学習など）が必要になります。
    # テンプレートを使用しない場合、この部分の精度は極めて低くなります。
    
    return "Unknown", 0.0


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
        min_line_length_px = st.sidebar.slider("線分と認識する最小の長さ (px)", 30, 200, 80) # 新しいパラメータ

        # --- 画像の二値化 ---
        blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        _, binarized_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        st.image(binarized_img, caption="二値化画像", use_column_width=True)

        st.subheader("ユーザー指定の画像処理ステップ")

        # --- Sobelフィルタ ---
        st.write("Sobelフィルタ適用")
        dx = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0)
        dy = cv2.Sobel(gray_img, cv2.CV_8U, 0, 1)
        # 浮動小数点型にキャストしてオーバーフローを防ぎ、0-255に正規化
        sobel_combined = np.sqrt(dx.astype(float) * dx.astype(float) + dy.astype(float) * dy.astype(float))
        # 最大値で割って0-255に正規化（ゼロ除算防止のため小さい値を加算）
        sobel_combined = (sobel_combined * 255.0 / (np.max(sobel_combined) + 1e-6)).astype('uint8') if np.max(sobel_combined) > 0 else np.zeros_like(sobel_combined, dtype='uint8')
        _, sobel_thresholded = cv2.threshold(sobel_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        st.image(sobel_thresholded, caption="Sobelフィルタと二値化", use_column_width=True)

        # --- Canny法 ---
        st.write("Canny法適用")
        canny = cv2.Canny(gray_img, 100, 200)
        st.image(canny, caption="Canny法によるエッジ検出", use_column_width=True)

        # --- モルフォロジー演算 ---
        st.write("モルフォロジー演算 (膨張・収縮) 適用 (二値化画像に対して)")
        kernel = np.ones((5, 5), dtype=np.uint8) # カーネルのデータ型をnp.uint8に指定
        res_dilate = cv2.dilate(binarized_img, kernel) # 既存の二値化画像 (binarized_img) を使用
        res_erode = cv2.erode(binarized_img, kernel)   # 既存の二値化画像 (binarized_img) を使用
        st.image(res_dilate, caption="膨張 (Dilation)", use_column_width=True)
        st.image(res_erode, caption="収縮 (Erosion)", use_column_width=True)

        # --- すべての輪郭を検出し、線分と記号を統一して分類 ---
        cnts, _ = detect_contours(gray_img, min_area, max_area)
        st.write(f"検出された輪郭数 (フィルタリング後): {len(cnts)}")

        detected_elements_info = []
        
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 非常に小さい外接矩形はスキップ
            if w < 5 or h < 5: 
                continue

            # 幾何学的特徴に基づいて要素を分類
            label, confidence = classify_element_by_geometry(cnt, src_img.shape[1], src_img.shape[0], min_line_length_px)

            if label == "Member_Line": # 部材線分は青で描画
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 2) # 青
                text = f"{label}"
                cv2.putText(output_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                draw_pca(output_img, cnt, (255, 0, 0)) # PCA軸も青
                detected_elements_info.append(f"- **{label}**: 位置 (x={x}, y={y}), 幅={w}, 高さ={h}")
            elif label != "Unknown": # 認識できた記号は緑で描画
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2) # 緑
                text = f"{label} (Conf: {confidence:.2f})"
                cv2.putText(output_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                draw_pca(output_img, cnt, (0, 255, 0)) # PCA軸も緑
                detected_elements_info.append(f"- **{label}**: 信頼度 {confidence:.2f}, 位置 (x={x}, y={y})")
            else: # 認識できなかったものは赤枠
                cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(output_img, "?", (x+2, y+h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        st.subheader("検出された要素（線分と記号）")
        if detected_elements_info:
            st.markdown("\n".join(detected_elements_info))
        else:
            st.write("要素は検出されませんでした。")
        
        st.markdown("""
        **幾何学的分類の注意点:**
        - **線分**は、その長さ、細さ、直線性に基づいて判別されます。手書きの揺らぎによっては、短い線分や曲がった線分が認識されない、または記号と誤認識される可能性があります。
        - **荷重 (矢印)** は複雑な形状のため、簡単な幾何学的特徴だけでは誤認識が多い可能性があります。
        - **ピン支点 (三角形)**, **ヒンジ (円)** は比較的判別しやすいですが、手書きのばらつきに左右されます。
        - **ピンローラー支点** や **固定支点** は、複数の図形要素の組み合わせで構成されるため、
          単一の輪郭の幾何学的特徴のみでは正確な判別は**非常に困難**です。
          これらの判別には、複数の輪郭間の空間的関係を分析する**より高度なアルゴリズム**や、
          パターン認識のための**機械学習モデルの訓練**が必要になります。
        """)

        st.subheader("最終結果画像")
        st.image(output_img, caption="検出結果", use_column_width=True)