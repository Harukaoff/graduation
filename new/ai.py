import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering # for node merging
import json

# --- 1. 画像の前処理 ---
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        st.error(f"画像ファイルが見つかりません: {img_path}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    median_blurred = cv2.medianBlur(gray, 5) 
    binary = cv2.adaptiveThreshold(median_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opening, img

# --- 2. 要素の検出（深層学習モデルのプレースホルダー） ---
def detect_structural_elements_with_dl(original_img_path):
    st.warning("このステップは現在、深層学習モデルの**ダミー検出結果**を生成しています。")
    st.info("ここに実際の訓練済み深層学習モデルの推論コードを組み込む必要があります。")

    dummy_img = cv2.imread(original_img_path)
    if dummy_img is None:
        return {"beams": [], "supports": [], "loads": [], "joints": []}

    h, w, _ = dummy_img.shape
    
    # ダミーの梁（より現実的に単一の水平梁に設定）
    dummy_beams = [
        ((int(w*0.15), int(h*0.5)), (int(w*0.85), int(h*0.5))),
    ]
    
    # ダミーの支点
    dummy_supports = [
        {"type": "pin_support", "coords": (int(w*0.15), int(h*0.5)), "confidence": 0.98},
        {"type": "roller_support", "coords": (int(w*0.85), int(h*0.5)), "confidence": 0.95}
    ]
    
    # ダミーの荷重
    dummy_loads = [
        {"type": "point_load", "coords": (int(w*0.5), int(h*0.4)), "magnitude": "5.0kN", "direction": "down", "confidence": 0.92}
    ]
    
    # ダミーのジョイント（梁の端点と荷重点のみに設定）
    dummy_joints = [
        {"coords": (int(w*0.15), int(h*0.5)), "confidence": 0.99},
        {"coords": (int(w*0.85), int(h*0.5)), "confidence": 0.99},
        {"coords": (int(w*0.5), int(h*0.4)), "confidence": 0.94} # 荷重がかかる点もジョイントとして認識
    ]
    
    return {
        "beams": dummy_beams,
        "supports": dummy_supports,
        "loads": dummy_loads,
        "joints": dummy_joints
    }

# --- 3. 構造体の認識・モデル化 ---
class StructureModel:
    def __init__(self):
        self.nodes = {}
        self.elements = []
        self.loads = []

def recognize_structure_from_dl_output(dl_results, merge_tolerance=20):
    model = StructureModel()
    node_id_counter = 0
    
    all_detected_points = []
    for beam in dl_results.get("beams", []):
        all_detected_points.append(beam[0])
        all_detected_points.append(beam[1])
    for joint in dl_results.get("joints", []):
        all_detected_points.append(joint['coords'])
    for support in dl_results.get("supports", []):
        all_detected_points.append(support['coords'])
    for load in dl_results.get("loads", []):
        all_detected_points.append(load['coords'])
    
    if all_detected_points:
        points_array = np.array(all_detected_points)
        
        if len(points_array) > 1:
            clustering = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=merge_tolerance)
            clustering.fit(points_array)
            labels = clustering.labels_
            
            merged_coords = []
            for i in range(clustering.n_clusters_):
                cluster_points = points_array[labels == i]
                merged_coords.append(tuple(np.mean(cluster_points, axis=0).astype(int)))
        else:
            merged_coords = [tuple(points_array[0].astype(int))] if len(points_array) == 1 else []

        for p_coords in merged_coords:
            model.nodes[node_id_counter] = {'coords': p_coords, 'type': 'joint'}
            node_id_counter += 1
            
    node_coords_to_id = {tuple(data['coords']): node_id for node_id, data in model.nodes.items()}
    
    added_elements = set()
    element_id_counter = 0
    for line_p1, line_p2 in dl_results.get("beams", []):
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
        
        if node_a_id is not None and node_b_id is not None and node_a_id != node_b_id \
           and min_dist_a < merge_tolerance * 1.5 and min_dist_b < merge_tolerance * 1.5:
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

    for support_candidate in dl_results.get("supports", []):
        s_cx, s_cy = support_candidate['coords']
        closest_node_id = None
        min_dist = float('inf')
        
        for node_id, node_data in model.nodes.items():
            n_x, n_y = node_data['coords']
            dist = np.linalg.norm(np.array([s_cx, s_cy]) - np.array([n_x, n_y]))
            if dist < min_dist:
                min_dist = dist
                closest_node_id = node_id
        
        if closest_node_id is not None and min_dist < merge_tolerance * 2:
            if model.nodes[closest_node_id]['type'] == 'joint':
                model.nodes[closest_node_id]['type'] = support_candidate['type']
            
    for load_candidate in dl_results.get("loads", []):
        l_cx, l_cy = load_candidate['coords']
        closest_node_id = None
        min_dist = float('inf')
        
        for node_id, node_data in model.nodes.items():
            n_x, n_y = node_data['coords']
            dist = np.linalg.norm(np.array([l_cx, l_cy]) - np.array([n_x, n_y]))
            if dist < min_dist:
                min_dist = dist
                closest_node_id = node_id
        
        if closest_node_id is not None and min_dist < merge_tolerance * 2:
            magnitude_val = float(load_candidate.get('magnitude', '10kN').replace('kN', ''))
            direction_val = load_candidate.get('direction', 'down')
            
            model.loads.append({
                'id': len(model.loads), 
                'target_id': closest_node_id, 
                'target_type': 'node', 
                'magnitude': magnitude_val, 
                'direction': direction_val
            })

    return model

# --- 4. 応力解析 (簡易版 - 変更なし) ---
class AnalysisResult:
    def __init__(self):
        self.shear_forces = {}
        self.bending_moments = {}
        self.axial_forces = {}
        self.reactions = {}

def perform_structural_analysis_simple(structure_model):
    results = AnalysisResult()
    for node_id, node in structure_model.nodes.items():
        if node['type'] == 'pin_support':
            results.reactions[node_id] = (np.random.rand() * 5, np.random.rand() * 15 + 5)
        elif node['type'] == 'roller_support':
            results.reactions[node_id] = (0.0, np.random.rand() * 15 + 5)
    
    for element in structure_model.elements:
        results.axial_forces[element['id']] = np.random.rand() * 5 - 2.5
        sf_start = np.random.rand() * 10 - 5
        sf_end = sf_start - np.random.rand() * 5
        results.shear_forces[element['id']] = [sf_start, sf_end]
        bm_start = np.random.rand() * 5
        bm_mid = np.random.rand() * 15 - 7.5
        bm_end = np.random.rand() * 5
        results.bending_moments[element['id']] = [bm_start, bm_mid, bm_end]
    return results

# --- 5. 製図化（応力図の生成） ---
def draw_stress_diagrams(original_img, structure_model, analysis_results, show_sfd=True, show_bmd=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("手書き構造体認識と応力図", fontsize=16)
    ax.axis('off')

    node_coords_map = {node_id: data['coords'] for node_id, data in structure_model.nodes.items()}
    
    for node_id, node_data in structure_model.nodes.items():
        x, y = node_data['coords']
        if node_data['type'] == 'joint':
            ax.plot(x, y, 'ko', markersize=5, zorder=3)

    for node_id, node_data in structure_model.nodes.items():
        x, y = node_data['coords']
        if node_data['type'] == 'pin_support':
            ax.plot(x, y + 15, 'v', color='blue', markersize=15, zorder=2, clip_on=False) 
            ax.plot([x-10, x+10], [y+15, y+15], 'b-', linewidth=2, zorder=2, clip_on=False)
        elif node_data['type'] == 'roller_support':
            circle = plt.Circle((x, y + 10), 5, color='green', fill=False, lw=2, zorder=2, clip_on=False)
            ax.add_patch(circle)
            ax.plot([x-10, x+10], [y+15, y+15], 'g-', linewidth=2, zorder=2, clip_on=False)
    
    for element in structure_model.elements:
        node_a_coords = node_coords_map[element['node_a_id']]
        node_b_coords = node_coords_map[element['node_b_id']]
        x_coords = [node_a_coords[0], node_b_coords[0]]
        y_coords = [node_a_coords[1], node_b_coords[1]]
        ax.plot(x_coords, y_coords, 'k-', linewidth=3, zorder=1)
        
        # 軸力の表示は常にオン (必要であればここもオプション化可能)
        if element['id'] in analysis_results.axial_forces:
            axial_force = analysis_results.axial_forces[element['id']]
            mid_x = (x_coords[0] + x_coords[1]) / 2
            mid_y = (y_coords[0] + y_coords[1]) / 2
            ax.text(mid_x, mid_y - 20, f"N={axial_force:.1f}", color='purple', 
                    fontsize=8, ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

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

    for node_id, reaction_vals in analysis_results.reactions.items():
        x, y = node_coords_map[node_id]
        Rx, Ry = reaction_vals
        arrow_length = 30
        head_width = 8
        head_length = 8
        
        if Ry > 0:
            ax.arrow(x + 20, y + arrow_length, 0, -(arrow_length - head_length), 
                     head_width=head_width, head_length=head_length, fc='darkorange', ec='darkorange', zorder=4)
            ax.text(x + 20, y + arrow_length + 10, f"Ry={Ry:.1f}", color='darkorange', ha='center', va='top', fontsize=10)
        else:
            ax.arrow(x + 20, y - arrow_length, 0, arrow_length - head_length, 
                     head_width=head_width, head_length=head_length, fc='darkorange', ec='darkorange', zorder=4)
            ax.text(x + 20, y - arrow_length - 10, f"Ry={Ry:.1f}", color='darkorange', ha='center', va='bottom', fontsize=10)
            
        if Rx > 0:
            ax.arrow(x - arrow_length, y + 20, arrow_length - head_length, 0, 
                     head_width=head_width, head_length=head_length, fc='darkorange', ec='darkorange', zorder=4)
            ax.text(x - arrow_length - 10, y + 20, f"Rx={Rx:.1f}", color='darkorange', ha='right', va='center', fontsize=10)
        else:
            ax.arrow(x + arrow_length, y + 20, -(arrow_length - head_length), 0, 
                     head_width=head_width, head_length=head_length, fc='darkorange', ec='darkorange', zorder=4)
            ax.text(x + arrow_length + 10, y + 20, f"Rx={Rx:.1f}", color='darkorange', ha='left', va='center', fontsize=10)

    # 応力図の描画 (せん断力図と曲げモーメント図) - オプションで表示/非表示を切り替え
    if show_sfd or show_bmd:
        for element_id, element in analysis_results.shear_forces.items(): # shear_forcesは全要素について存在する前提
            el = next(e for e in structure_model.elements if e['id'] == element_id)
            node_a_coords = node_coords_map[el['node_a_id']]
            node_b_coords = node_coords_map[el['node_b_id']]

            vec = np.array(node_b_coords) - np.array(node_a_coords)
            length = np.linalg.norm(vec)
            if length == 0: continue
            
            normal_vec = np.array([-vec[1], vec[0]]) / length
            if normal_vec[1] < 0:
                 normal_vec = -normal_vec
            
            # せん断力図 (青い破線)
            if show_sfd:
                sf_vals = analysis_results.shear_forces[element['id']]
                sf_start_offset = np.array(node_a_coords) + normal_vec * sf_vals[0] * 3
                sf_end_offset = np.array(node_b_coords) + normal_vec * sf_vals[1] * 3
                ax.plot([sf_start_offset[0], sf_end_offset[0]], 
                        [sf_start_offset[1], sf_end_offset[1]], 
                        'b--', linewidth=1.5, label='Shear Force' if element_id == 0 else None, zorder=2)
                ax.text((sf_start_offset[0] + sf_end_offset[0]) / 2, 
                        (sf_start_offset[1] + sf_end_offset[1]) / 2 + 10, 
                        "SFD", color='blue', fontsize=8, ha='center', va='bottom')

            # 曲げモーメント図 (緑の点線)
            if show_bmd:
                bm_vals = analysis_results.bending_moments[element['id']]
                bm_start_offset = np.array(node_a_coords) - normal_vec * bm_vals[0] * 3
                bm_mid_offset = np.array(node_a_coords) + vec / 2 - normal_vec * bm_vals[1] * 3
                bm_end_offset = np.array(node_b_coords) - normal_vec * bm_vals[2] * 3
                
                ax.plot([bm_start_offset[0], bm_mid_offset[0], bm_end_offset[0]],
                        [bm_start_offset[1], bm_mid_offset[1], bm_end_offset[1]],
                        'g-.', linewidth=1.5, label='Bending Moment' if element_id == 0 else None, zorder=2)
                ax.text((bm_start_offset[0] + bm_mid_offset[0] + bm_end_offset[0]) / 3, 
                        (bm_start_offset[1] + bm_mid_offset[1] + bm_end_offset[1]) / 3 - 10, 
                        "BMD", color='green', fontsize=8, ha='center', va='top')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    st.pyplot(fig)

# --- Streamlit アプリケーションの構築 ---
st.set_page_config(layout="wide", page_title="手書き構造体応力図アプリ")

st.title("🏗️ 手書き構造体応力図生成アプリ (深層学習モデル連携版)")

st.write("""
このアプリは、手書きで描かれた構造図から**深層学習モデルが検出した要素**を基に構造モデルを構築し、応力図を生成します。
現在、**深層学習モデルの検出結果はダミー**です。ここに実際の訓練済みモデルを組み込む必要があります。
""")

uploaded_file = st.file_uploader("手書きの構造図をアップロードしてください (PNG, JPG)", type=["png", "jpg", "jpeg"])

# サイドバーにオプションを追加
st.sidebar.header("表示オプション")
show_sfd_option = st.sidebar.checkbox("せん断力図 (SFD) を表示", value=False) # デフォルトでOFF
show_bmd_option = st.sidebar.checkbox("曲げモーメント図 (BMD) を表示", value=False) # デフォルトでOFF

if uploaded_file is not None:
    temp_image_path = "temp_image.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="アップロードされた画像", use_column_width=True)
    st.write("---")

    st.header("ステップ 1: 画像の前処理")
    binary_img, original_img = preprocess_image(temp_image_path)
    
    if binary_img is not None:
        st.image(binary_img, caption="二値化された画像 (参考)", use_column_width=True, channels="GRAY")
        
        st.header("ステップ 2: 構造要素の検出 (深層学習モデルによる)")
        st.info("これは現在ダミーの検出結果です。実際の深層学習モデルに置き換えてください。")
        dl_detection_results = detect_structural_elements_with_dl(temp_image_path)
        
        st.write("### 深層学習モデルによる検出結果 (ダミー)")
        st.json(dl_detection_results)
        
        display_img_detections = original_img.copy()
        for beam in dl_detection_results.get("beams", []):
            cv2.line(display_img_detections, beam[0], beam[1], (0, 255, 0), 2)
        for support in dl_detection_results.get("supports", []):
            cv2.circle(display_img_detections, support['coords'], 8, (255, 0, 255), -1)
            cv2.putText(display_img_detections, support['type'], (support['coords'][0]+10, support['coords'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        for load in dl_detection_results.get("loads", []):
            cv2.circle(display_img_detections, load['coords'], 8, (0, 255, 255), -1)
            cv2.putText(display_img_detections, f"{load.get('magnitude', '')} {load.get('direction', '')}", 
                        (load['coords'][0]+10, load['coords'][1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for joint in dl_detection_results.get("joints", []):
            cv2.circle(display_img_detections, joint['coords'], 5, (255, 165, 0), -1)
            
        st.image(cv2.cvtColor(display_img_detections, cv2.COLOR_BGR2RGB), 
                 caption="深層学習モデルによる検出結果の可視化 (ダミー)", use_column_width=True)

        st.header("ステップ 3: 構造体の認識・モデル化")
        merge_tolerance = st.slider("節点マージの許容誤差 (ピクセル)", 5, 50, 20)
        structure_model = recognize_structure_from_dl_output(dl_detection_results, merge_tolerance=merge_tolerance)
        
        st.write("### 認識された構造モデル")
        st.json({
            "nodes": {k: v for k, v in structure_model.nodes.items()},
            "elements": structure_model.elements,
            "loads": structure_model.loads
        })

        if structure_model.nodes and structure_model.elements:
            st.header("ステップ 4: 応力解析 (FEM連携予定)")
            analysis_results = perform_structural_analysis_simple(structure_model)
            st.write("### 解析結果 (簡易版)")
            st.json({
                "shear_forces": analysis_results.shear_forces,
                "bending_moments": analysis_results.bending_moments,
                "axial_forces": analysis_results.axial_forces,
                "reactions": analysis_results.reactions
            })
            
            st.header("ステップ 5: 応力図の生成")
            # 応力図の表示オプションを渡す
            draw_stress_diagrams(original_img, structure_model, analysis_results, 
                                 show_sfd=show_sfd_option, show_bmd=show_bmd_option)
            st.success("応力図が生成されました！")
        else:
            st.warning("検出された要素が少なすぎるか、適切に構造体をモデル化できませんでした。深層学習モデルの精度向上、またはマージ閾値の調整を検討してください。")
    else:
        st.error("画像の読み込みまたは前処理に失敗しました。")

st.info("""
**今後の具体的な進め方:**

1.  **データセットの構築:**
    * 様々な手書きの梁、柱、ピン支点、ローラー支点、固定支点、集中荷重（点荷重）、等分布荷重、偶力などを描いた画像を収集します。
    * これらの要素に対して、**バウンディングボックス**や**セグメンテーションマスク**、そして**クラスラベル**（例: "beam", "pin_support", "point_load", "distributed_load" など）、可能であれば**荷重値や方向などの属性**をアノテーションします。
    * アノテーションツールとして、LabelImg（バウンディングボックス）、Labelme（セグメンテーション）などが利用できます。

2.  **深層学習モデルの選定と訓練:**
    * **オブジェクト検出 (Object Detection)**: YOLO (You Only Look Once), Faster R-CNN, EfficientDet など。これらのモデルは、各要素のバウンディングボックスとクラスを検出するのに適しています。荷重値や方向のOCRも組み合わせることも考えられます。
    * **セマンティックセグメンテーション (Semantic Segmentation)**: U-Net, DeepLab など。ピクセル単位で梁、支点、荷重の領域を識別するのに適しています。これにより、要素の形状をより正確に把握できます。
    * **グラフニューラルネットワーク (GNN)**: 検出された要素間の接続関係（トポロジー）を学習・推論するために利用できますが、これはより高度なステップです。まずはオブジェクト検出/セグメンテーションから始めると良いでしょう。
    * **訓練**: 準備したデータセットを用いて、選定したモデルを訓練します。PythonでPyTorchやTensorFlow、Kerasなどのフレームワークを使用します。Colab ProやAWS/GCPのGPUインスタンスを利用すると効率的です。

3.  **モデルの組み込み:**
    * 訓練済みモデルをONNX形式などで保存し、Streamlitアプリから読み込めるようにします。
    * `detect_structural_elements_with_dl` 関数の内部を、**実際のモデルの推論コード**に置き換えます。モデルの出力形式に合わせて、`dl_detection_results` 辞書の形式に変換するロジックを記述します。

4.  **FEMコードとの連携:**
    * `perform_structural_analysis_simple` 関数を、用意されている**FEMコードを呼び出すロジック**に置き換えます。
    * `StructureModel` オブジェクトから、FEMコードが必要とする入力形式（節点座標、要素接続、荷重ベクトル、拘束条件など）にデータを変換するアダプター層を実装します。

5.  **UIのさらなる改善:**
    * 検出結果をユーザーが確認し、**必要に応じて手動で修正・追加・削除できる**インタラクティブなUIを実装します。例えば、検出された荷重の値を編集したり、支点のタイプを変更したり、梁の接続を修正したりする機能です。これはStreamlitの描画機能と、ユーザーからの入力を組み合わせることで実現できます。
""")