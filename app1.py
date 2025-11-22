import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import math

# fem_lib.pyをインポートするためにパスを追加
# Streamlitが実行される環境に合わせてパスを調整してください
# sys.path.append(os.path.dirname(__file__))
from fem_lib import esm, d_r

st.set_page_config(layout="wide") # レイアウトを広めに設定

st.title("構造図からの要素認識と動的FEMシミュレーション")

st.header("1. 構造要素の画像認識")

# --- テンプレートの読み込み ---
template_files = {
    "ピン": "templates/pin2.png",
    "ローラー": "templates/roller2.png",
    "固定": "templates/fixed1.png",
    "ヒンジ": "templates/hinge.png",
    "荷重": "templates/kajyu.png"
}
templates = {}
for label, path in template_files.items():
    try:
        templ = cv2.imread(path, 0)
        if templ is not None:
            templates[label] = templ
        else:
            st.warning(f"テンプレート画像が見つからないか、読み込めません: {path}")
    except Exception as e:
        st.error(f"テンプレート画像読み込みエラー {path}: {e}")

if not templates:
    st.error("テンプレート画像が一つも読み込めませんでした。'templates'フォルダ内の画像ファイルを確認してください。")

# --- 輪郭検出関数 ---
def detect_contours(gray_img):
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if 100 < cv2.contourArea(cnt) < 5000], thr

# --- PCA描画関数 ---
def draw_pca(img_draw, cnt_data):
    if len(cnt_data) < 2:
        return img_draw
    data = cnt_data.reshape(-1, 2).astype(np.float32)
    try:
        mean, eigen = cv2.PCACompute(data, mean=None)
        center = tuple(mean[0].astype(int))
        direction = eigen[0] * 100
        end = (int(center[0] + direction[0]), int(center[1] + direction[1]))
        cv2.line(img_draw, center, end, (255, 0, 0), 2)
    except cv2.error as e:
        pass # エラーは頻繁に出るため表示しない
    return img_draw

# --- テンプレートマッチング関数 ---
def match_template_region(roi_img):
    best_label, best_score = None, -1
    for label, templ in templates.items():
        if templ is None:
            continue
        try:
            if roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
                continue
            roi_resize = cv2.resize(roi_img, (templ.shape[1], templ.shape[0]))
            res = cv2.matchTemplate(roi_resize, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            if score > best_score:
                best_label, best_score = label, score
        except cv2.error as e:
            pass
    return best_label, best_score

# --- Streamlit アプリのメイン処理 (画像認識部分) ---
uploaded_file = st.file_uploader("構造図の画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file:
    buf = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    src_img = cv2.imdecode(buf, 1)
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    output_img = src_img.copy()

    cnts, thr_img = detect_contours(gray_img)
    st.write(f"検出された輪郭数: {len(cnts)}")
    st.image(thr_img, caption='二値化画像', use_column_width=True)

    detected_elements_info = []
    
    match_threshold = st.slider("マッチングの信頼度閾値", 0.0, 1.0, 0.5, 0.05)

    count = 0
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue

        roi = gray_img[y:y+h, x:x+w]
        label, score = match_template_region(roi)

        if label and score > match_threshold:
            count += 1
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} {score:.2f} ({x},{y})"
            cv2.putText(output_img, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            draw_pca(output_img, cnt)
            detected_elements_info.append(f"- **{label}**: 信頼度 {score:.2f}, 位置 (x={x}, y={y})")
        else:
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(output_img, "?", (x+2, y+h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB),
              caption=f"検出された構造要素: {count}個", use_column_width=True)

    if detected_elements_info:
        st.write("### 検出された要素の詳細:")
        for info in detected_elements_info:
            st.markdown(info)
    else:
        st.info("画像から有効な構造要素は見つかりませんでした。テンプレートや画像のコントラスト、または閾値を確認してください。")

st.markdown("---")
st.header("2. FEM解析の入力設定 (手動で動的に設定)")
st.write("画像認識の結果を参考に、以下のフォームで構造体の詳細を入力してください。")

# --- FEM解析パラメータの手動設定 ---
st.subheader("部材共通パラメータ")
col_E, col_A, col_I = st.columns(3)
E_val = col_E.number_input("ヤング率 E (N/mm²)", value=2.0 * 10**5, format="%.2e")
A_val = col_A.number_input("断面積 A (mm²)", value=6.0 * 10**3, format="%.2e")
I_val = col_I.number_input("断面二次モーメント I (mm⁴)", value=2.0 * 10**7, format="%.2e")

st.subheader("部材の定義")
num_elements = st.number_input("部材の数", min_value=1, max_value=10, value=2, step=1)

element_inputs = []
max_node_idx = 0

for i in range(num_elements):
    st.markdown(f"#### 部材 {i+1}")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    start_node = col1.number_input(f"始点ノード ({i+1})", key=f"start_node_{i}", value=i)
    end_node = col2.number_input(f"終点ノード ({i+1})", key=f"end_node_{i}", value=i+1)
    length = col3.number_input(f"長さ L (mm) ({i+1})", key=f"length_{i}", value=4000)
    angle = col4.number_input(f"角度 (度) ({i+1})", key=f"angle_{i}", value=0 if i % 2 == 0 else 270) # デフォルトで水平/鉛直
    Ws = col5.number_input(f"分布荷重 始点 Ws (N/mm) ({i+1})", key=f"Ws_{i}", value=0.0)
    We = col6.number_input(f"分布荷重 終点 We (N/mm) ({i+1})", key=f"We_{i}", value=0.0)
    
    element_inputs.append({
        "start": start_node,
        "end": end_node,
        "length": length,
        "angle": angle,
        "Ws": Ws,
        "We": We
    })
    max_node_idx = max(max_node_idx, start_node, end_node)

st.subheader("節点データ (nodes_df) の定義")
st.warning(
    "各節点の拘束条件 (rc_x, rc_y, rc_m) と外力 (ef_x, ef_y, ef_m) をPythonの辞書形式で入力してください。\n"
    f"節点番号は 0 から {max_node_idx} まで考慮する必要があります。\n"
    "入力例を参考に、必要な節点のリストの長さを調整してください。"
)

# L字型フレームのnodes_df定義例 (ノード0:固定, ノード1:中間, ノード2:ピン+荷重)
default_nodes_data_str = f"""{{
    'rc_x': [1, 0, 1]{', 0' * (max_node_idx - 2) if max_node_idx > 1 else ''},
    'rc_y': [1, 0, 1]{', 0' * (max_node_idx - 2) if max_node_idx > 1 else ''},
    'rc_m': [1, 0, 0]{', 0' * (max_node_idx - 2) if max_node_idx > 1 else ''},
    'ef_x': [0, 0, 0]{', 0' * (max_node_idx - 2) if max_node_idx > 1 else ''},
    'ef_y': [0, 0, -1000]{', 0' * (max_node_idx - 2) if max_node_idx > 1 else ''}, # ノード2に下向き1000N
    'ef_m': [0, 0, 0]{', 0' * (max_node_idx - 2) if max_node_idx > 1 else ''}
}}"""

# 節点データをテキストエリアで入力
nodes_data_str = st.text_area(
    "nodes_df のデータ (Python辞書形式)",
    value=default_nodes_data_str,
    height=250
)

# --- 3. FEM解析の実行 ---
st.header("3. FEM解析の実行")
if st.button("FEM解析を実行"):
    st.write("FEM解析を実行中...")
    
    # element_list_fem を動的に構築
    element_list_fem = []
    for elem_data in element_inputs:
        K_element = esm(E_val, A_val, I_val, elem_data["length"], elem_data["angle"])
        element_list_fem.append((
            K_element, 
            elem_data["start"], 
            elem_data["end"], 
            elem_data["angle"], 
            elem_data["Ws"], 
            elem_data["We"], 
            elem_data["length"]
        ))
    
    # nodes_df_fem をテキストエリアからパース
    try:
        nodes_data_fem = eval(nodes_data_str) # ユーザー入力を辞書として評価
        nodes_df_fem = pd.DataFrame(nodes_data_fem)
        
        # 節点番号が nodes_df_fem の行数と一致しているか簡易チェック
        if len(nodes_df_fem) <= max_node_idx:
            st.error(f"nodes_dfの行数（現在{len(nodes_df_fem)}）が、最大節点番号（{max_node_idx}）に対応していません。リストの長さを確認してください。")
            st.stop() # 処理を中断
        
    except SyntaxError as e:
        st.error(f"nodes_dfの入力形式が不正です。Pythonの辞書形式で正しく入力してください。エラー: {e}")
        st.stop()
    except Exception as e:
        st.error(f"nodes_dfの処理中に予期せぬエラーが発生しました: {e}")
        st.stop()

    # FEM解析の実行
    try:
        result_fem = d_r(element_list_fem, nodes_df_fem)

        if result_fem is not None:
            st.write("### FEM解析結果")
            st.dataframe(result_fem) 
            
            st.write("#### 主要な結果:")
            # 解析結果の各項目が結果DataFrameに含まれるかチェックして表示
            result_indices = result_fem.index.tolist()
            # 変位
            for i in range(max_node_idx + 1):
                if f'u{i}' in result_indices:
                    st.write(f"ノード{i} 水平変位 (u{i}): {result_fem.loc[f'u{i}'].values[0]:.4f} mm")
                if f'v{i}' in result_indices:
                    st.write(f"ノード{i} 鉛直変位 (v{i}): {result_fem.loc[f'v{i}'].values[0]:.4f} mm")
                if f'theta{i}' in result_indices:
                    st.write(f"ノード{i} 回転角 (theta{i}): {result_fem.loc[f'theta{i}'].values[0]:.4f} rad")
            
            # 反力
            for i in range(max_node_idx + 1):
                 if f'px{i}' in result_indices:
                    st.write(f"ノード{i} 反力X (px{i}): {result_fem.loc[f'px{i}'].values[0]:.4f} N")
                 if f'py{i}' in result_indices:
                    st.write(f"ノード{i} 反力Y (py{i}): {result_fem.loc[f'py{i}'].values[0]:.4f} N")
                 if f'M{i}' in result_indices:
                    st.write(f"ノード{i} 反力モーメント (M{i}): {result_fem.loc[f'M{i}'].values[0]:.4f} N·mm")

        else:
            st.error("FEM解析に失敗しました。構造が不安定であるか、入力に問題がある可能性があります。")
    except Exception as e:
        st.error(f"FEM解析中に予期せぬエラーが発生しました: {e}")

st.markdown("---")
st.info("""
**今後の課題:**

1.  **幾何情報（節点座標、部材の長さ・角度・接続）の自動抽出:** 現在の画像認識は記号の検出までです。図面から部材の線分や節点の位置を特定し、それらの接続関係（トポロジー）を自動で構築する機能が、FEM解析への完全な自動入力には不可欠です。
2.  **荷重の数値の認識:** 荷重記号だけでなく、その横に書かれた「1000N」といった数値を光学文字認識（OCR）で読み取る必要があります。
3.  **ユーザーによる確認・修正インターフェース:** 画像認識の誤りをユーザーが簡単に修正できるUI（ドラッグ＆ドロップ、数値入力欄など）を設けることが重要です。
4.  **結果の可視化:** 解析結果（変位図、応力図など）を視覚的に表示する機能を追加することで、より直感的な理解が可能になります。
""")