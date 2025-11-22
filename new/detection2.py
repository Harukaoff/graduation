import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance # 点間の距離計算に利用

# --- 1. 画像の前処理 ---
def preprocess_image(img_path):
    """
    手書き画像を読み込み、グレースケール化、ノイズ除去、二値化を行う。
    """
    img = cv2.imread(img_path)
    if img is None:
        st.error(f"画像ファイルが見つかりません: {img_path}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去: メディアンフィルターを適用（点状ノイズに強い）
    median_blurred = cv2.medianBlur(gray, 5) # カーネルサイズ5
    
    # 適応的閾値処理で線の太さや濃淡のばらつきに対応
    # blockSize と C の値は画像によって調整が必要
    binary = cv2.adaptiveThreshold(median_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # モルフォロジー変換（オープニング）で小さいノイズを除去し、線を滑らかにする
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opening, img # 処理後の二値画像と元のカラー画像を返す

# --- 2. 要素の検出（梁、支点、荷重） ---
def detect_structural_elements(binary_img, line_threshold, min_line_length, max_line_gap):
    """
    二値化画像から梁、支点、荷重をそれぞれ検出する。
    簡易的なルールベースの検出。
    """
    height, width = binary_img.shape
    
    # 梁の検出（Hough変換）
    # threshold: 線として認識する際の最小投票数。小さいほど多くの線（ノイズ含む）を検出。
    # minLineLength: 検出する線の最小長さ。短いノイズ線を除去。
    # maxLineGap: 線を繋げる最大の隙間。大きいほど途切れた線を繋げやすい。
    lines = cv2.HoughLinesP(binary_img, 1, np.pi / 180, 
                            threshold=line_threshold, 
                            minLineLength=min_line_length, 
                            maxLineGap=max_line_gap)

    detected_beams = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_beams.append(((x1, y1), (x2, y2)))
            
    # 支点と荷重の検出（輪郭分析と形状特徴に基づく簡易版）
    # これらの検出は手書き図形の多様性に非常に弱いです。
    # 深層学習によるオブジェクト検出が理想的です。
    
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_supports = []
    detected_loads = []

    for cnt in contours:
        # 小さすぎる領域はノイズとして無視
        if cv2.contourArea(cnt) < 50: # 面積の閾値は調整可能
            continue
        
        # 凸包を計算（外形を単純化）
        hull = cv2.convexHull(cnt)
        
        # 輪郭の近似
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 境界ボックスとアスペクト比、ソリディティ（面積/凸包面積）
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        # オブジェクトの中心座標
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2 # 面積が小さい場合

        # --- 支点検出の簡易ロジック ---
        # 三角形っぽい形状 (ピン支点)
        if len(approx) == 3 and aspect_ratio > 0.5 and aspect_ratio < 1.5 and solidity > 0.8:
            # 最下部にある三角形をピン支点として優先
            if cy > height * 0.7: # 画像の下部にある程度限定
                detected_supports.append({'type': 'pin_support_candidate', 'coords': (cx, cy)})
                
        # 円形っぽい形状 (ローラー支点)
        elif len(approx) > 5 and solidity > 0.8 and aspect_ratio > 0.7 and aspect_ratio < 1.3:
            # 最下部にある円をローラー支点として優先
            if cy > height * 0.7: # 画像の下部にある程度限定
                detected_supports.append({'type': 'roller_support_candidate', 'coords': (cx, cy)})

        # --- 荷重検出の簡易ロジック（矢印） ---
        # 矢印は複雑なので、ここでは単純に細長い形状で、かつ直線ではないものを候補とする
        # または、輪郭の直線と三角形の組み合わせを検出する高度なロジックが必要
        # ここでは、アスペクト比が極端に大きい/小さい、かつ、細長いが完全に直線ではないものを候補とする
        # 完璧ではないが、ヒューリスティックな試み
        if (aspect_ratio > 3 or aspect_ratio < 0.33) and area > 100 and solidity < 0.9:
            # 矢印の先端を推定する簡易ロジック（最も外側に突き出た点など）
            # ここでは中心座標をとりあえず使う
            detected_loads.append({'type': 'load_candidate', 'coords': (cx, cy)})

    return detected_beams, detected_supports, detected_loads

# --- 3. 構造体の認識・モデル化 ---
class StructureModel:
    def __init__(self):
        self.nodes = {} # {id: {'coords': (x, y), 'type': 'joint'/'pin_support'/'roller_support'}}
        self.elements = [] # [{'id': id, 'node_a_id': id_a, 'node_b_id': id_b, 'type': 'beam'}]
        self.loads = [] # [{'id': id, 'target_id': node_id, 'magnitude': val, 'direction': 'down'}]

def recognize_structure(detected_beams, detected_supports, detected_loads, merge_tolerance=20):
    """
    検出された梁、支点、荷重候補から構造モデルを構築。
    """
    model = StructureModel()
    node_id_counter = 0
    
    # 梁の端点を節点として初期登録・マージ
    for line_p1, line_p2 in detected_beams:
        for p in [line_p1, line_p2]:
            found_existing_node = False
            for existing_node_id, existing_node_data in model.nodes.items():
                existing_p = existing_node_data['coords']
                if np.linalg.norm(np.array(p) - np.array(existing_p)) < merge_tolerance:
                    found_existing_node = True
                    break
            
            if not found_existing_node:
                model.nodes[node_id_counter] = {'coords': p, 'type': 'joint'}
                node_id_counter += 1

    # 要素（梁）の作成
    added_elements = set()
    element_id_counter = 0
    for line_p1, line_p2 in detected_beams:
        node_a_id = None
        node_b_id = None
        min_dist_a = float('inf')
        min_dist_b = float('inf')

        # 梁の端点に最も近い節点を見つける
        for node_id, node_data in model.nodes.items():
            dist_p1_to_node = np.linalg.norm(np.array(line_p1) - np.array(node_data['coords']))
            if dist_p1_to_node < min_dist_a:
                min_dist_a = dist_p1_to_node
                node_a_id = node_id
            
            dist_p2_to_node = np.linalg.norm(np.array(line_p2) - np.array(node_data['coords']))
            if dist_p2_to_node < min_dist_b:
                min_dist_b = dist_p2_to_node
                node_b_id = node_id
        
        if node_a_id is not None and node_b_id is not None and node_a_id != node_b_id:
            sorted_nodes = tuple(sorted((node_a_id, node_b_id)))
            if sorted_nodes not in added_elements:
                model.elements.append({
                    'id': element_id_counter, 
                    'node_a_id': node_a_id, 
                    'node_b_id': node_b_id, 
                    'type': 'beam'
                })
                added_elements.add(sorted_nodes)
                element_id_counter += 1

    # 支点の割り当て
    for support_candidate in detected_supports:
        s_cx, s_cy = support_candidate['coords']
        closest_node_id = None
        min_dist = float('inf')
        
        for node_id, node_data in model.nodes.items():
            n_x, n_y = node_data['coords']
            dist = np.linalg.norm(np.array([s_cx, s_cy]) - np.array([n_x, n_y]))
            if dist < min_dist:
                min_dist = dist
                closest_node_id = node_id
        
        # 近くに節点があれば、その節点を支点として設定
        if closest_node_id is not None and min_dist < merge_tolerance * 1.5: # 支点と節点間の許容誤差
            model.nodes[closest_node_id]['type'] = support_candidate['type'].replace('_candidate', '')
            
    # 荷重の割り当て
    for load_candidate in detected_loads:
        l_cx, l_cy = load_candidate['coords']
        closest_node_id = None
        min_dist = float('inf')
        
        for node_id, node_data in model.nodes.items():
            n_x, n_y = node_data['coords']
            dist = np.linalg.norm(np.array([l_cx, l_cy]) - np.array([n_x, n_y]))
            if dist < min_dist:
                min_dist = dist
                closest_node_id = node_id
        
        # 近くに節点があれば、その節点に荷重を設定
        if closest_node_id is not None and min_dist < merge_tolerance * 1.5: # 荷重と節点間の許容誤差
            # 現時点では、全ての荷重を10kNの下向きと仮定
            model.loads.append({
                'id': len(model.loads), 
                'target_id': closest_node_id, 
                'target_type': 'node', 
                'magnitude': 10.0, 
                'direction': 'down'
            })

    return model


# --- 4. 応力解析 (簡易版) ---
class AnalysisResult:
    def __init__(self):
        self.shear_forces = {}    # {element_id: [start_val, end_val], ...}
        self.bending_moments = {} # {element_id: [start_val, mid_val, end_val], ...}
        self.axial_forces = {}    # {element_id: value, ...}
        self.reactions = {}       # {node_id: (Rx, Ry), ...}

def perform_structural_analysis_simple(structure_model):
    """
    構造モデルと荷重情報に基づいて応力解析を実行する簡易的な関数。
    非常に単純なケース（例：単純梁など）を想定。本格的な解析には別途ライブラリや詳細な実装が必要。
    """
    results = AnalysisResult()

    # 仮の反力と応力値を設定
    # 実際の解析では、支点と荷重の組み合わせに基づいて計算が必要
    
    # 反力の計算（非常に簡易的）
    for node_id, node in structure_model.nodes.items():
        if node['type'] == 'pin_support':
            results.reactions[node_id] = (np.random.rand() * 5, np.random.rand() * 15 + 5) # ダミーの反力
        elif node['type'] == 'roller_support':
            results.reactions[node_id] = (0.0, np.random.rand() * 15 + 5) # ダミーの反力 (水平反力なし)
    
    # 各要素の応力計算（非常に簡易的）
    for element in structure_model.elements:
        results.axial_forces[element['id']] = np.random.rand() * 5 - 2.5 # ダミーの軸力
        
        # 梁に何らかの荷重がかかっていると仮定して、せん断力と曲げモーメントを生成
        # 例：単純梁に中央集中荷重や等分布荷重がかかった場合を模倣
        sf_start = np.random.rand() * 10 - 5
        sf_end = sf_start - np.random.rand() * 5 # 適当な変化
        results.shear_forces[element['id']] = [sf_start, sf_end]

        bm_start = np.random.rand() * 5
        bm_mid = np.random.rand() * 15 - 7.5
        bm_end = np.random.rand() * 5
        results.bending_moments[element['id']] = [bm_start, bm_mid, bm_end] # 開始、中央、終了

    return results

# --- 5. 製図化（応力図の生成） ---
def draw_stress_diagrams(original_img, structure_model, analysis_results):
    """
    元の画像の上に、認識された構造体と応力図を描画する。
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 元の画像を背景に表示
    ax.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("手書き構造体認識と応力図", fontsize=16)
    ax.axis('off') # 軸を非表示に

    # 構造体の描画
    # まずノードを描画して、座標辞書を構築
    node_coords_map = {node_id: data['coords'] for node_id, data in structure_model.nodes.items()}
    
    for node_id, node_data in structure_model.nodes.items():
        x, y = node_data['coords']
        # 支点でない場合は通常の節点として描画
        if node_data['type'] == 'joint':
            ax.plot(x, y, 'ko', markersize=5, zorder=3) # 節点

    # 支点の描画（ノードとは別に描画することで、アイコンの制御を容易にする）
    for node_id, node_data in structure_model.nodes.items():
        x, y = node_data['coords']
        if node_data['type'] == 'pin_support':
            # ピン支点（三角形）
            ax.plot(x, y + 15, 'v', color='blue', markersize=15, zorder=2, clip_on=False) 
            ax.plot([x-10, x+10], [y+15, y+15], 'b-', linewidth=2, zorder=2, clip_on=False) # 地面
        elif node_data['type'] == 'roller_support':
            # ローラー支点（円と線）
            circle = plt.Circle((x, y + 10), 5, color='green', fill=False, lw=2, zorder=2, clip_on=False)
            ax.add_patch(circle)
            ax.plot([x-10, x+10], [y+15, y+15], 'g-', linewidth=2, zorder=2, clip_on=False) # 地面
    
    # 要素（梁）の描画
    for element in structure_model.elements:
        node_a_coords = node_coords_map[element['node_a_id']]
        node_b_coords = node_coords_map[element['node_b_id']]
        
        x_coords = [node_a_coords[0], node_b_coords[0]]
        y_coords = [node_a_coords[1], node_b_coords[1]]
        
        ax.plot(x_coords, y_coords, 'k-', linewidth=3, zorder=1) # 梁
        
        # 軸力の表示
        if element['id'] in analysis_results.axial_forces:
            axial_force = analysis_results.axial_forces[element['id']]
            mid_x = (x_coords[0] + x_coords[1]) / 2
            mid_y = (y_coords[0] + y_coords[1]) / 2
            ax.text(mid_x, mid_y - 20, f"N={axial_force:.1f}", color='purple', 
                    fontsize=8, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 荷重の描画
    for load in structure_model.loads:
        if load['target_type'] == 'node':
            x, y = node_coords_map[load['target_id']]
            arrow_length = 30
            head_width = 8
            head_length = 8
            
            if load['direction'] == 'down':
                ax.arrow(x, y - arrow_length, 0, arrow_length - head_length, 
                         head_width=head_width, head_length=head_length, fc='red', ec='red', zorder=4)
                ax.text(x, y - arrow_length - 10, f"{load['magnitude']:.1f}kN", color='red', ha='center', va='bottom', fontsize=10)
            # 他の方向の荷重もここに追加可能

    # 反力の描画
    for node_id, reaction_vals in analysis_results.reactions.items():
        x, y = node_coords_map[node_id]
        Rx, Ry = reaction_vals
        arrow_length = 30
        head_width = 8
        head_length = 8
        
        # Ry (鉛直反力)
        if Ry > 0: # 上向き
            ax.arrow(x + 20, y + arrow_length, 0, -(arrow_length - head_length), 
                     head_width=head_width, head_length=head_length, fc='darkorange', ec='darkorange', zorder=4)
            ax.text(x + 20, y + arrow_length + 10, f"Ry={Ry:.1f}", color='darkorange', ha='center', va='top', fontsize=10)
        else: # 下向き
            ax.arrow(x + 20, y - arrow_length, 0, arrow_length - head_length, 
                     head_width=head_width, head_length=head_length, fc='darkorange', ec='darkorange', zorder=4)
            ax.text(x + 20, y - arrow_length - 10, f"Ry={Ry:.1f}", color='darkorange', ha='center', va='bottom', fontsize=10)
            
        # Rx (水平反力)
        if Rx > 0: # 右向き
            ax.arrow(x - arrow_length, y + 20, arrow_length - head_length, 0, 
                     head_width=head_width, head_length=head_length, fc='darkorange', ec='darkorange', zorder=4)
            ax.text(x - arrow_length - 10, y + 20, f"Rx={Rx:.1f}", color='darkorange', ha='right', va='center', fontsize=10)
        else: # 左向き
            ax.arrow(x + arrow_length, y + 20, -(arrow_length - head_length), 0, 
                     head_width=head_width, head_length=head_length, fc='darkorange', ec='darkorange', zorder=4)
            ax.text(x + arrow_length + 10, y + 20, f"Rx={Rx:.1f}", color='darkorange', ha='left', va='center', fontsize=10)

    # 応力図の描画 (せん断力図と曲げモーメント図)
    # 各要素の長さに応じて応力図の形状を調整
    for element_id, element in analysis_results.shear_forces.items():
        el = next(e for e in structure_model.elements if e['id'] == element_id)
        node_a_coords = node_coords_map[el['node_a_id']]
        node_b_coords = node_coords_map[el['node_b_id']]

        # 梁のベクトル
        vec = np.array(node_b_coords) - np.array(node_a_coords)
        length = np.linalg.norm(vec)
        if length == 0: continue
        
        # 梁に垂直な単位ベクトル (応力図を描画する方向)
        normal_vec = np.array([-vec[1], vec[0]]) / length # 時計回りに90度回転
        # 必要であれば、normal_vecの向きを反転させて、常に梁の下側に描画
        if normal_vec[1] < 0: # y成分が負なら上向きなので反転
             normal_vec = -normal_vec
        
        # せん断力図 (青い破線)
        sf_vals = analysis_results.shear_forces[element_id]
        # 開始点と終了点の応力図オフセット座標
        sf_start_offset = np.array(node_a_coords) + normal_vec * sf_vals[0] * 3 # スケール調整
        sf_end_offset = np.array(node_b_coords) + normal_vec * sf_vals[1] * 3
        ax.plot([sf_start_offset[0], sf_end_offset[0]], 
                [sf_start_offset[1], sf_end_offset[1]], 
                'b--', linewidth=1.5, label='Shear Force' if element_id == 0 else None, zorder=2)
        ax.text((sf_start_offset[0] + sf_end_offset[0]) / 2, 
                (sf_start_offset[1] + sf_end_offset[1]) / 2 + 10, # 少しずらして文字が重ならないように
                "SFD", color='blue', fontsize=8, ha='center', va='bottom')

        # 曲げモーメント図 (緑の点線)
        bm_vals = analysis_results.bending_moments[element_id]
        # 開始点、中央点、終了点の応力図オフセット座標
        bm_start_offset = np.array(node_a_coords) - normal_vec * bm_vals[0] * 3
        bm_mid_offset = np.array(node_a_coords) + vec / 2 - normal_vec * bm_vals[1] * 3
        bm_end_offset = np.array(node_b_coords) - normal_vec * bm_vals[2] * 3
        
        # 放物線ではなく直線で近似して描画
        ax.plot([bm_start_offset[0], bm_mid_offset[0], bm_end_offset[0]],
                [bm_start_offset[1], bm_mid_offset[1], bm_end_offset[1]],
                'g-.', linewidth=1.5, label='Bending Moment' if element_id == 0 else None, zorder=2)
        ax.text((bm_start_offset[0] + bm_mid_offset[0] + bm_end_offset[0]) / 3, 
                (bm_start_offset[1] + bm_mid_offset[1] + bm_end_offset[1]) / 3 - 10, # 少しずらして文字が重ならないように
                "BMD", color='green', fontsize=8, ha='center', va='top')
    
    # 重複ラベルを避けるため、一度だけ凡例を表示
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    st.pyplot(fig) # Streamlitでmatplotlibの図を表示

# --- Streamlit アプリケーションの構築 ---
st.set_page_config(layout="wide", page_title="手書き構造体応力図アプリ")

st.title("🏗️ 手書き構造体応力図生成アプリ")

st.write("""
このアプリは、手書きで描かれた構造図から**梁（直線）**、**支点（三角形、円）**、**荷重（矢印）**のそれぞれの要素を検出し、
それらを基に簡易的な応力図を生成します。
""")

uploaded_file = st.file_uploader("手書きの構造図をアップロードしてください (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # ファイルを一時的に保存
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="アップロードされた画像", use_column_width=True)
    st.write("---")

    st.header("ステップ 1: 画像の前処理")
    binary_img, original_img = preprocess_image("temp_image.png")
    
    if binary_img is not None:
        st.image(binary_img, caption="二値化された画像", use_column_width=True, channels="GRAY")
        
        st.header("ステップ 2: 構造要素の検出")
        st.subheader("検出パラメータの調整")
        line_threshold = st.slider("梁のHough変換閾値", 10, 200, 50)
        min_line_length = st.slider("梁の最小長さ", 10, 100, 50)
        max_line_gap = st.slider("梁の最大隙間", 0, 50, 10)

        detected_beams, detected_supports, detected_loads = detect_structural_elements(
            binary_img, line_threshold, min_line_length, max_line_gap
        )
        
        if detected_beams or detected_supports or detected_loads:
            st.write(f"検出された梁の数: {len(detected_beams)}")
            st.write(f"検出された支点候補の数: {len(detected_supports)}")
            st.write(f"検出された荷重候補の数: {len(detected_loads)}")
            
            # 検出結果の可視化
            display_img_detections = original_img.copy()
            for line_coords in detected_beams:
                cv2.line(display_img_detections, line_coords[0], line_coords[1], (0, 255, 0), 2) # 緑色で梁
            for support in detected_supports:
                cv2.circle(display_img_detections, support['coords'], 8, (255, 0, 255), -1) # マゼンタで支点候補
            for load in detected_loads:
                cv2.circle(display_img_detections, load['coords'], 8, (0, 255, 255), -1) # シアンで荷重候補
            
            st.image(cv2.cvtColor(display_img_detections, cv2.COLOR_BGR2RGB), 
                     caption="検出された梁、支点、荷重候補", use_column_width=True)

            st.header("ステップ 3: 構造体の認識・モデル化")
            merge_tolerance = st.slider("梁端点・要素結合の許容誤差 (ピクセル)", 5, 50, 20)
            structure_model = recognize_structure(
                detected_beams, detected_supports, detected_loads, merge_tolerance=merge_tolerance
            )
            
            st.write("### 認識された構造モデル")
            st.json({
                "nodes": {k: v for k, v in structure_model.nodes.items()},
                "elements": structure_model.elements,
                "loads": structure_model.loads
            })

            if structure_model.nodes and structure_model.elements:
                st.header("ステップ 4: 応力解析")
                analysis_results = perform_structural_analysis_simple(structure_model)
                st.write("### 解析結果 (簡易版)")
                st.json({
                    "shear_forces": analysis_results.shear_forces,
                    "bending_moments": analysis_results.bending_moments,
                    "axial_forces": analysis_results.axial_forces,
                    "reactions": analysis_results.reactions
                })
                
                st.header("ステップ 5: 応力図の生成")
                draw_stress_diagrams(original_img, structure_model, analysis_results)
                st.success("応力図が生成されました！")
            else:
                st.warning("検出された要素が少なすぎるか、適切に構造体をモデル化できませんでした。")
        else:
            st.warning("画像から構造要素が検出されませんでした。パラメータ調整や、より鮮明な画像をお試しください。")
    else:
        st.error("画像の読み込みまたは前処理に失敗しました。")

st.info("""
**今後の改善点:**
* **高精度な構造体認識:** 現状は簡易なルールベース検出です。手書きの多様なスタイルや複雑な形状に対応するため、**深層学習（セグメンテーションモデル、オブジェクト検出モデル、グラフニューラルネットワークなど）**を導入することが不可欠です。これにより、支点や荷重の形状をより正確に認識し、荷重値や支点タイプを文字認識で自動判別できるようになります。
* **正確な構造解析:** 現在の応力解析はダミーです。正確な解析を行うには、**有限要素法 (FEM) などの数値解析アルゴリズム**を実装するか、既存の構造解析ライブラリと連携させる必要があります。これにより、不静定構造など、より複雑な構造の解析も可能になります。
* **ユーザーインタフェースの改善:** 荷重の数値や方向、支点のタイプ（ピン、ローラー、固定など）を**ユーザーが直接入力・修正**できる機能を追加すると、柔軟性が増します。
* **手書き文字認識 (OCR) の統合:** 荷重の数値や単位、部材の名称などを画像から自動で読み取るために、**手書き文字認識（OCR）技術**を組み込む必要があります。これは、深層学習モデルを用いたアプローチが強力です。
""")