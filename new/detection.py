import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
# networkx は直接的なグラフ構築には不要になったが、将来的な拡張のために残すことも可能
# from scipy.spatial import distance # 点間の距離計算にはまだ利用

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

# --- 2. 線の検出 ---
def detect_lines_only_simple(binary_img):
    """
    二値化画像からHough変換を用いて線分のみを検出する。
    """
    # HoughLinesP のパラメータ調整が重要
    # threshold: 線として認識する際の最小投票数。小さいほど多くの線（ノイズ含む）を検出。
    # minLineLength: 検出する線の最小長さ。短いノイズ線を除去。
    # maxLineGap: 線を繋げる最大の隙間。大きいほど途切れた線を繋げやすい。
    lines = cv2.HoughLinesP(binary_img, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    detected_lines_coords = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines_coords.append(((x1, y1), (x2, y2)))

    return detected_lines_coords

# --- 3. 構造体の認識・モデル化 (線のみに基づく簡易版) ---
class StructureModel:
    def __init__(self):
        self.nodes = {} # {id: {'coords': (x, y), 'type': 'joint'/'pin_support'/'roller_support'}}
        self.elements = [] # [{'id': id, 'node_a_id': id_a, 'node_b_id': id_b, 'type': 'beam'}]
        self.loads = [] # [{'id': id, 'target_id': node_id, 'magnitude': val, 'direction': 'up'/'down'/'left'/'right'}]

def recognize_structure_from_lines(detected_lines, merge_tolerance=20):
    """
    検出された線分から、その端点を節点とし、線分を要素（梁）とする簡易的な構造モデルを構築。
    近接する端点は同じ節点としてマージする。
    """
    model = StructureModel()
    node_id_counter = 0
    
    # すでに登録された節点の座標とIDのリスト
    registered_nodes_coords = []
    
    # 線の端点をループし、既存の節点とマージするか、新しい節点として追加
    for line_p1, line_p2 in detected_lines:
        for p in [line_p1, line_p2]:
            found_existing_node = False
            for existing_node_id, existing_node_data in model.nodes.items():
                existing_p = existing_node_data['coords']
                # 既存の節点と十分近い場合、その節点にマージ
                if np.linalg.norm(np.array(p) - np.array(existing_p)) < merge_tolerance:
                    found_existing_node = True
                    break
            
            if not found_existing_node:
                # 新しい節点として追加
                model.nodes[node_id_counter] = {'coords': p, 'type': 'joint'}
                node_id_counter += 1

    # 節点IDと座標のマッピングを更新
    node_coords_to_id = {tuple(data['coords']): node_id for node_id, data in model.nodes.items()}

    # 要素（梁）の作成
    element_id_counter = 0
    added_elements = set() # 重複要素を避けるためのセット
    
    for line_p1, line_p2 in detected_lines:
        # 線の端点に最も近い節点を見つける
        node_a_id = None
        node_b_id = None
        min_dist_a = float('inf')
        min_dist_b = float('inf')

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
            # 重複要素を避けるために、タプルで正規化してセットに追加
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

    # 支点の自動付与（最も下にある節点を支点とする簡易ロジック）
    # より現実的な設定として、X座標が異なる2点を支点にする
    if model.nodes:
        # すべてのノードをY座標（高さ）で降順にソート（下にあるほどYが大きい）
        sorted_nodes_by_y = sorted(model.nodes.items(), key=lambda item: item[1]['coords'][1], reverse=True)
        
        support_candidates = []
        if sorted_nodes_by_y:
            # 最も下にあるノード群の中から候補を選ぶ
            bottom_y_threshold = sorted_nodes_by_y[0][1]['coords'][1] - 30 # 最下点から30px以内を「下部」とする
            for node_id, node_data in sorted_nodes_by_y:
                if node_data['coords'][1] >= bottom_y_threshold:
                    support_candidates.append((node_id, node_data['coords']))
            
            if len(support_candidates) >= 2:
                # 候補の中からX座標でソートし、両端を支点とする
                support_candidates.sort(key=lambda x: x[1][0]) # X座標でソート
                
                pin_support_id = support_candidates[0][0]
                roller_support_id = support_candidates[-1][0]
                
                model.nodes[pin_support_id]['type'] = 'pin_support'
                if pin_support_id != roller_support_id: # 同じノードでなければローラー支点も設定
                    model.nodes[roller_support_id]['type'] = 'roller_support'
            elif len(support_candidates) == 1: # 候補が1つだけの場合
                 model.nodes[support_candidates[0][0]]['type'] = 'pin_support'


    # 荷重の自動付与（最も上にある節点に下向きの荷重を付与）
    if model.nodes:
        # すべてのノードをY座標（高さ）で昇順にソート（上にあるほどYが小さい）
        sorted_nodes_by_y = sorted(model.nodes.items(), key=lambda item: item[1]['coords'][1])
        if sorted_nodes_by_y:
            top_node_id = sorted_nodes_by_y[0][0]
            # 既に支点となっている節点には荷重を付与しない
            if model.nodes[top_node_id]['type'] == 'joint': 
                model.loads.append({
                    'id': 0, 
                    'target_id': top_node_id, 
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
        ax.plot(x, y, 'ko', markersize=5, zorder=3) # 節点
        
        # 支点タイプに応じてアイコンを変える
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
        # 基本的に線の下側（Y座標が大きい方向）に描くように調整
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
このアプリは、手書きで描かれた構造図から**線分**を検出し、それを基に簡易的な応力図を生成します。
不要な点ノイズの影響を減らすため、点検出は行っていません。
現在、構造体の認識と応力解析は非常にシンプルなロジックに基づいており、複雑な図形や正確な解析には対応していません。
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
        
        st.header("ステップ 2: 線の検出")
        detected_lines = detect_lines_only_simple(binary_img)
        
        if detected_lines:
            st.write(f"検出された線分の数: {len(detected_lines)}")
            
            # 検出結果の可視化
            display_img_detections = original_img.copy()
            for line_coords in detected_lines:
                cv2.line(display_img_detections, line_coords[0], line_coords[1], (0, 255, 0), 2) # 緑色で線
            
            st.image(cv2.cvtColor(display_img_detections, cv2.COLOR_BGR2RGB), 
                     caption="検出された線", use_column_width=True)

            st.header("ステップ 3: 構造体の認識・モデル化")
            # 線の端点をマージする閾値
            merge_tolerance = st.slider("線の端点マージ許容誤差 (ピクセル)", 5, 50, 20)
            structure_model = recognize_structure_from_lines(detected_lines, merge_tolerance=merge_tolerance)
            
            st.write("### 認識された構造モデル")
            # デバッグ用にモデルの内容を表示
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
                st.warning("検出された線分が少なすぎるか、適切に構造体をモデル化できませんでした。")
        else:
            st.warning("画像から線が検出されませんでした。より鮮明な画像や、Hough変換のパラメータ調整をお試しください。")
    else:
        st.error("画像の読み込みまたは前処理に失敗しました。")

st.info("""
**今後の改善点:**
* **高精度な構造体認識:** 深層学習（セグメンテーションモデル、オブジェクト検出モデル、グラフニューラルネットワークなど）を導入し、手書きの歪みや複雑な図形にも対応できるようにする。
* **正確な構造解析:** より汎用的な構造解析アルゴリズム（有限要素法など）を実装または既存ライブラリと連携させる。
* **ユーザーインタフェースの改善:** 荷重の入力、支点タイプの選択など、ユーザーが構造条件を細かく設定できる機能を追加する。
* **手書き文字認識 (OCR) の統合:** 荷重の数値や単位などを自動で読み取る機能を追加する。
""")