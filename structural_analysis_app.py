import os
import math
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import fem_lib
import draw_lib

def solve_frame(nodes_df, elements_df):
    """FEM解析を実行する統合関数"""
    # データフォーマットを変換
    elements_converted = []
    nodes_converted = []
    
    # 要素データを変換
    for _, element in elements_df.iterrows():
        # 節点座標を取得
        node_i = nodes_df[nodes_df['node'] == element['node_i']].iloc[0]
        node_j = nodes_df[nodes_df['node'] == element['node_j']].iloc[0]
        
        # 長さと角度を計算
        length = np.sqrt((node_j['x'] - node_i['x'])**2 + (node_j['y'] - node_i['y'])**2)
        angle = math.degrees(math.atan2(node_j['y'] - node_i['y'], node_j['x'] - node_i['x']))
        if angle < 0:
            angle += 360
        
        elements_converted.append([
            element['E'],      # young
            element['A'],      # area  
            element['I'],      # s_moment
            length,           # length
            angle,            # angle
            element['node_i'], # start
            element['node_j'], # end
            0,                # Ws (分布荷重開始)
            0                 # We (分布荷重終了)
        ])
    
    # 節点データを変換
    for _, node in nodes_df.iterrows():
        nodes_converted.append([
            node['x'],        # x座標
            node['y'],        # y座標
            1 if node['fix_x'] else 0,    # rc_x
            1 if node['fix_y'] else 0,    # rc_y
            1 if node['fix_theta'] else 0, # rc_m
            node['Fx'],       # ef_x
            node['Fy'],       # ef_y
            node['M']         # ef_m
        ])
    
    # DataFrameに変換
    elements_fem_df = pd.DataFrame(elements_converted, columns=[
        'young', 'area', 's_moment', 'length', 'angle', 'start', 'end', 'Ws', 'We'
    ])
    
    nodes_fem_df = pd.DataFrame(nodes_converted, columns=[
        'x', 'y', 'rc_x', 'rc_y', 'rc_m', 'ef_x', 'ef_y', 'ef_m'
    ])
    
    # FEM解析実行
    displacements, member_stresses = fem_lib.fem_calc(elements_fem_df, nodes_fem_df)
    
    return displacements, member_stresses

def draw_deformation(nodes_df, elements_df, displacements, scale=50, show_original=True):
    """変形図を描画"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 元の構造を描画
    if show_original:
        for _, element in elements_df.iterrows():
            node_i = nodes_df[nodes_df['node'] == element['node_i']].iloc[0]
            node_j = nodes_df[nodes_df['node'] == element['node_j']].iloc[0]
            ax.plot([node_i['x'], node_j['x']], [node_i['y'], node_j['y']], 
                   'k--', linewidth=1, alpha=0.5, label='元形状' if element.name == 0 else "")
    
    # 変形後の構造を描画
    for _, element in elements_df.iterrows():
        node_i_idx = element['node_i']
        node_j_idx = element['node_j']
        
        # 元の座標
        node_i = nodes_df[nodes_df['node'] == node_i_idx].iloc[0]
        node_j = nodes_df[nodes_df['node'] == node_j_idx].iloc[0]
        
        # 変形後の座標
        disp_i = displacements.iloc[node_i_idx]
        disp_j = displacements.iloc[node_j_idx]
        
        x_i_def = node_i['x'] + disp_i['u'] * scale
        y_i_def = node_i['y'] + disp_i['v'] * scale
        x_j_def = node_j['x'] + disp_j['u'] * scale
        y_j_def = node_j['y'] + disp_j['v'] * scale
        
        ax.plot([x_i_def, x_j_def], [y_i_def, y_j_def], 
               'r-', linewidth=2, label='変形後' if element.name == 0 else "")
    
    # 節点を描画
    for _, node in nodes_df.iterrows():
        disp = displacements.iloc[node['node']]
        x_def = node['x'] + disp['u'] * scale
        y_def = node['y'] + disp['v'] * scale
        
        # 拘束条件に応じて色を変更
        if node['fix_x'] and node['fix_y'] and node['fix_theta']:
            color = 'red'  # 固定端
        elif node['fix_x'] and node['fix_y']:
            color = 'orange'  # ピン支点
        elif node['fix_y']:
            color = 'green'  # ローラー支点
        else:
            color = 'blue'  # 自由端
        
        ax.plot(x_def, y_def, 'o', color=color, markersize=6)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'変形図 (倍率: {scale}倍)', fontsize=14)
    ax.legend()
    ax.invert_y()
    
    return fig

def draw_stress_diagram(nodes_df, elements_df, member_stresses):
    """応力図を描画"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 軸力図
    ax1.set_title('軸力図 (N)', fontsize=12)
    for i, stress_df in enumerate(member_stresses):
        ax1.plot(stress_df['delta'], stress_df['N'], 'b-', linewidth=2, label=f'部材{i}')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('軸力 N')
    
    # せん断力図
    ax2.set_title('せん断力図 (Q)', fontsize=12)
    for i, stress_df in enumerate(member_stresses):
        ax2.plot(stress_df['delta'], stress_df['Q'], 'g-', linewidth=2, label=f'部材{i}')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('せん断力 Q')
    
    # 曲げモーメント図
    ax3.set_title('曲げモーメント図 (M)', fontsize=12)
    for i, stress_df in enumerate(member_stresses):
        ax3.plot(stress_df['delta'], stress_df['M'], 'r-', linewidth=2, label=f'部材{i}')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('曲げモーメント M')
    ax3.set_xlabel('部材軸方向距離')
    
    plt.tight_layout()
    return fig

# ===== 構造グラフ生成クラス =====
class Node:
    """構造解析用の節点クラス"""
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.fix_x = False
        self.fix_y = False
        self.fix_theta = False
        self.loads = np.zeros(3)  # [Fx, Fy, M]
    
    def __repr__(self):
        return f"Node({self.id}, {self.x:.1f}, {self.y:.1f})"

class Element:
    """構造解析用の要素クラス"""
    def __init__(self, id, n1, n2, E=2.0e2, A=9.0e2, I=6.75e4):
        self.id = id
        self.n1 = n1  # 開始節点ID
        self.n2 = n2  # 終了節点ID
        self.E = E    # ヤング係数
        self.A = A    # 断面積
        self.I = I    # 断面二次モーメント
    
    def __repr__(self):
        return f"Element({self.id}, {self.n1}-{self.n2})"

class StructureGraph:
    """構造グラフ管理クラス"""
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.node_counter = 0
        self.element_counter = 0
    
    def add_node(self, x, y):
        """節点を追加"""
        node = Node(self.node_counter, x, y)
        self.nodes.append(node)
        self.node_counter += 1
        return node
    
    def find_or_add_node(self, x, y, tol=15):
        """既存節点を検索、なければ新規追加"""
        for node in self.nodes:
            if np.hypot(node.x - x, node.y - y) < tol:
                return node
        return self.add_node(x, y)
    
    def add_element(self, n1_id, n2_id, E=2.0e2, A=9.0e2, I=6.75e4):
        """要素を追加"""
        element = Element(self.element_counter, n1_id, n2_id, E, A, I)
        self.elements.append(element)
        self.element_counter += 1
        return element
    
    def visualize(self):
        """構造グラフを可視化"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 要素を描画
        for element in self.elements:
            n1 = self.nodes[element.n1]
            n2 = self.nodes[element.n2]
            ax.plot([n1.x, n2.x], [n1.y, n2.y], 'b-', linewidth=3, alpha=0.7)
            
            # 要素番号を表示
            mid_x = (n1.x + n2.x) / 2
            mid_y = (n1.y + n2.y) / 2
            ax.text(mid_x, mid_y, f'E{element.id}', fontsize=8, ha='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # 節点を描画
        for node in self.nodes:
            # 拘束条件に応じて色を変更
            if node.fix_x and node.fix_y and node.fix_theta:
                color = 'red'  # 固定端
                marker = 's'
            elif node.fix_x and node.fix_y:
                color = 'orange'  # ピン支点
                marker = 'o'
            elif node.fix_y:
                color = 'green'  # ローラー支点
                marker = '^'
            else:
                color = 'blue'  # 自由端
                marker = 'o'
            
            ax.plot(node.x, node.y, marker=marker, color=color, markersize=8)
            ax.text(node.x, node.y + 15, f'N{node.id}', fontsize=10, ha='center', fontweight='bold')
            
            # 荷重を表示
            if np.any(node.loads != 0):
                ax.arrow(node.x, node.y, node.loads[0]*5, node.loads[1]*5, 
                        head_width=5, head_length=5, fc='red', ec='red')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('構造グラフ', fontsize=14, fontweight='bold')
        ax.invert_y()  # 画像座標系に合わせる
        
        return fig

def create_simple_beam_structure(supports, loads, structure_graph):
    """単純梁構造を生成"""
    if len(supports) < 2:
        raise ValueError("単純梁には最低2つの支点が必要です")
    
    # 支点を左右に並べ替え
    supports_sorted = sorted(supports, key=lambda s: s['node'][0])
    
    # 2つの支点節点を追加
    node1 = structure_graph.add_node(supports_sorted[0]['node'][0], supports_sorted[0]['node'][1])
    node2 = structure_graph.add_node(supports_sorted[1]['node'][0], supports_sorted[1]['node'][1])
    
    # 支点の拘束条件を設定
    for i, support in enumerate([supports_sorted[0], supports_sorted[1]]):
        node = node1 if i == 0 else node2
        support_type = support['type']
        
        if support_type == "pin":
            node.fix_x = True
            node.fix_y = True
        elif support_type == "roller":
            node.fix_y = True
        elif support_type == "fixed":
            node.fix_x = True
            node.fix_y = True
            node.fix_theta = True
    
    # 2つの支点を結ぶ梁要素を追加
    structure_graph.add_element(node1.id, node2.id)
    
    # 荷重を梁上に配置
    beam_length = np.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)
    
    for load in loads:
        load_center = load['pts'].mean(axis=0)
        
        # 梁上の最も近い点を計算
        t = np.dot([load_center[0] - node1.x, load_center[1] - node1.y], 
                  [node2.x - node1.x, node2.y - node1.y]) / (beam_length**2)
        t = max(0.1, min(0.9, t))  # 端点から少し離す
        
        # 荷重作用点の節点を追加
        load_x = node1.x + t * (node2.x - node1.x)
        load_y = node1.y + t * (node2.y - node1.y)
        load_node = structure_graph.add_node(load_x, load_y)
        
        # 荷重を設定
        if load['type'] == 'load':
            # 集中荷重（下向き）
            load_node.loads[1] = -10.0  # 下向きを負とする
        elif load['type'] in ['momentl', 'momentr']:
            # モーメント荷重
            sign = -1 if load['type'] == 'momentl' else 1
            load_node.loads[2] = sign * 10.0
    
    return structure_graph

def beam_pts_to_line(pts):
    """
    OBBの4点から梁の中心線を取得
    """
    # 長辺を探す
    dists = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        dists.append((np.linalg.norm(p2-p1), p1, p2))
    
    dists.sort(key=lambda x: x[0], reverse=True)
    _, p1, p2 = dists[0]
    return p1, p2

def build_elements_from_beams(beams, structure_graph):
    """梁検出結果から要素を生成"""
    for beam in beams:
        p1, p2 = beam_pts_to_line(beam["pts"])
        
        n1 = structure_graph.find_or_add_node(p1[0], p1[1])
        n2 = structure_graph.find_or_add_node(p2[0], p2[1])
        
        if n1.id != n2.id:
            structure_graph.add_element(n1.id, n2.id)

def apply_supports_to_nodes(supports, structure_graph, tol=20):
    """支点を節点に適用"""
    for support in supports:
        sx, sy = support["node"]
        for node in structure_graph.nodes:
            if np.hypot(node.x - sx, node.y - sy) < tol:
                if support["type"] == "pin":
                    node.fix_x = True
                    node.fix_y = True
                elif support["type"] == "roller":
                    node.fix_y = True
                elif support["type"] == "fixed":
                    node.fix_x = True
                    node.fix_y = True
                    node.fix_theta = True

def apply_loads_to_nodes(loads, structure_graph, tol=20):
    """荷重を節点に適用"""
    for load in loads:
        load_center = load['pts'].mean(axis=0)
        
        # 最も近い節点を見つける
        min_dist = float('inf')
        nearest_node = None
        for node in structure_graph.nodes:
            dist = np.hypot(node.x - load_center[0], node.y - load_center[1])
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        if nearest_node and min_dist < tol:
            if load['type'] == 'load':
                nearest_node.loads[1] = -10.0  # 下向き
            elif load['type'] in ['momentl', 'momentr']:
                sign = -1 if load['type'] == 'momentl' else 1
                nearest_node.loads[2] = sign * 10.0

def create_frame_structure(beams, supports, loads, structure_graph):
    """ラーメン構造を生成（検出された梁配置をそのまま使用）"""
    # 梁から要素を生成
    build_elements_from_beams(beams, structure_graph)
    
    # 支点を節点に適用
    apply_supports_to_nodes(supports, structure_graph)
    
    # 荷重を節点に適用
    apply_loads_to_nodes(loads, structure_graph)
    
    return structure_graph

def validate_structure(structure_graph):
    """構造の妥当性をチェック"""
    if len(structure_graph.elements) == 0:
        raise ValueError("梁要素が生成されていません")
    if not any(n.fix_x or n.fix_y for n in structure_graph.nodes):
        raise ValueError("支点が設定されていません")
    if len(structure_graph.nodes) < 2:
        raise ValueError("節点数が不足しています（最低2個必要）")

# Streamlit設定
st.set_page_config(
    layout="wide", 
    page_title="InstaStruct - 構造力学解析アプリ",
    page_icon="🏗️"
)

# 設定
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r"C:\Users\morim\Downloads\graduation\runs\obb\train31\weights\best.pt"
if not os.path.exists(TEMPLATE_DIR):
    TEMPLATE_DIR = r"C:\Users\morim\Downloads\graduation\templates"

MODEL_PATH = os.getenv("MODEL_PATH", MODEL_PATH)
TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", TEMPLATE_DIR)

support_types = {"pin", "roller", "fixed", "hinge"}
load_types = {"load", "udl", "momentl", "momentr"}

# タイトル
st.title("InstaStruct - 構造解析アプリ")
st.write("手書き構造図から自動で構造解析を行い、変形図と応力図を出力します")

# サイドバー設定
with st.sidebar:
    st.header("⚙️ 解析設定")
    
    # 構造タイプの選択
    st.subheader("🏗️ 構造タイプ")
    structure_type = st.radio(
        "解析する構造を選択してください:",
        ["単純梁", "ラーメン構造"],
        help="単純梁: 2つの支点を直線で結ぶ\nラーメン構造: 検出された梁の配置をそのまま使用"
    )

# 画像アップロード
uploaded = st.file_uploader("📷 構造図画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("画像ファイルをアップロードしてください")
    st.stop()

# 画像処理
img_pil = Image.open(uploaded).convert("RGB")
img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

col1, col2 = st.columns(2)
with col1:
    st.image(img_pil, caption="元画像", use_container_width=True)

# モデル読み込み
if not os.path.exists(MODEL_PATH):
    st.error(f"モデルパスが存在しません: {MODEL_PATH}")
    st.stop()

if st.button("🚀 解析実行", type="primary"):
    with st.spinner("画像認識中..."):
        model = YOLO(MODEL_PATH)
        res = model(img, conf=0.5, imgsz=640)[0]
    
    # 検出結果を解析
    obb = res.obb
    supports, beams, loads = [], [], []
    
    if hasattr(obb, "xyxyxyxy"):
        N = len(obb.xyxyxyxy)
        for i in range(N):
            conf = float(obb.conf[i])
            if conf < 0.5: continue
            cls_id = int(obb.cls[i])
            name = res.names[cls_id].lower().replace(" ", "")
            pts = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
            
            if name in support_types:
                node = pts.mean(axis=0)
                supports.append(dict(type=name, node=node, pts=pts, conf=conf))
            elif name == "beam":
                beams.append({"type": "beam", "pts": pts, "conf": conf})
            elif name in load_types:
                loads.append({"type": name, "pts": pts, "conf": conf})
    
    # 構造グラフを生成
    structure_graph = StructureGraph()
    
    try:
        if structure_type == "単純梁":
            structure_graph = create_simple_beam_structure(supports, loads, structure_graph)
        else:  # ラーメン構造
            structure_graph = create_frame_structure(beams, supports, loads, structure_graph)
        
        # 構造の妥当性をチェック
        validate_structure(structure_graph)
        
        # 構造グラフを可視化
        with col2:
            st.subheader("🏗️ 生成された構造グラフ")
            fig = structure_graph.visualize()
            st.pyplot(fig, use_container_width=True)
        
        # 構造情報を表示
        with st.expander("📋 構造詳細情報"):
            st.write(f"**節点数**: {len(structure_graph.nodes)}")
            st.write(f"**要素数**: {len(structure_graph.elements)}")
            
            st.write("**節点一覧**:")
            for node in structure_graph.nodes:
                constraints = []
                if node.fix_x: constraints.append("x固定")
                if node.fix_y: constraints.append("y固定")
                if node.fix_theta: constraints.append("θ固定")
                constraint_str = ", ".join(constraints) if constraints else "自由"
                
                loads_str = f"荷重: ({node.loads[0]:.1f}, {node.loads[1]:.1f}, {node.loads[2]:.1f})"
                st.write(f"  - {node} | {constraint_str} | {loads_str}")
            
            st.write("**要素一覧**:")
            for element in structure_graph.elements:
                st.write(f"  - {element}")
        
        # FEM解析実行
        st.subheader("🔬 構造解析結果")
        
        with st.spinner("FEM解析実行中..."):
            # 構造グラフをFEMライブラリ用のデータに変換
            nodes_data = []
            elements_data = []
            
            # 節点データを準備
            for node in structure_graph.nodes:
                nodes_data.append([
                    node.id,
                    node.x,
                    node.y,
                    1 if node.fix_x else 0,
                    1 if node.fix_y else 0,
                    1 if node.fix_theta else 0,
                    node.loads[0],  # Fx
                    node.loads[1],  # Fy
                    node.loads[2]   # M
                ])
            
            # 要素データを準備
            for element in structure_graph.elements:
                elements_data.append([
                    element.id,
                    element.n1,
                    element.n2,
                    element.E,
                    element.A,
                    element.I
                ])
            
            # DataFrameに変換
            nodes_df = pd.DataFrame(nodes_data, columns=[
                'node', 'x', 'y', 'fix_x', 'fix_y', 'fix_theta', 'Fx', 'Fy', 'M'
            ])
            elements_df = pd.DataFrame(elements_data, columns=[
                'element', 'node_i', 'node_j', 'E', 'A', 'I'
            ])
            
            try:
                # FEM解析実行
                displacements, member_stresses = solve_frame(nodes_df, elements_df)
                
                # 結果を可視化
                col3, col4 = st.columns(2)
                
                with col3:
                    st.write("**🔄 変形図**")
                    deform_fig = draw_deformation(
                        nodes_df, elements_df, displacements, 
                        scale=50, show_original=True
                    )
                    if deform_fig:
                        # 軸と枠線を削除
                        deform_fig.gca().set_xticks([])
                        deform_fig.gca().set_yticks([])
                        deform_fig.gca().spines['top'].set_visible(False)
                        deform_fig.gca().spines['right'].set_visible(False)
                        deform_fig.gca().spines['bottom'].set_visible(False)
                        deform_fig.gca().spines['left'].set_visible(False)
                        st.pyplot(deform_fig, use_container_width=True)
                
                with col4:
                    st.write("**📊 応力図**")
                    stress_fig = draw_stress_diagram(
                        nodes_df, elements_df, member_stresses
                    )
                    if stress_fig:
                        # 軸と枠線を削除
                        for ax in stress_fig.get_axes():
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['bottom'].set_visible(False)
                            ax.spines['left'].set_visible(False)
                        st.pyplot(stress_fig, use_container_width=True)
                
                # 数値結果を表示
                with st.expander("📊 詳細な解析結果"):
                    st.write("**変位結果**:")
                    st.dataframe(displacements)
                    st.write("**応力結果**:")
                    for i, stress_df in enumerate(member_stresses):
                        st.write(f"**部材 {i} の応力**:")
                        st.dataframe(stress_df)
                
                st.success("✅ 構造解析完了")
                
            except Exception as fem_error:
                st.error(f"❌ FEM解析エラー: {str(fem_error)}")
                st.write("**デバッグ情報**:")
                st.write("節点データ:")
                st.dataframe(nodes_df)
                st.write("要素データ:")
                st.dataframe(elements_df)
                st.exception(fem_error)
        
    except Exception as e:
        st.error(f"❌ 構造グラフ生成エラー: {str(e)}")
        st.exception(e)