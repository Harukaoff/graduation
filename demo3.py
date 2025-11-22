import sys
import os
# fem_lib.py が同じディレクトリにあることを確認
sys.path.append(os.path.dirname(__file__)) 
from fem_lib import esm, d_r # fem_libから必要な関数をインポート
import streamlit as st
import pandas as pd
import numpy as np

st.title("小規模 FEM 実行デモ")

# 部材パラメータ
# 現実的な鋼材のヤング率 (N/mm^2) に修正
E = 2.0 * 10**5 # 200 GPa = 200 * 10^3 N/mm^2
A = 6.0 * 10**3 # 断面積 (mm^2)
I = 2.0 * 10**7 # 断面二次モーメント (mm^4)
L = 8000        # 長さ (mm)
angle = 0       # 角度 (度) - 水平方向

# 局所剛性行列 (esmは要素の全体座標系剛性行列を返します)
K_element = esm(E, A, I, L, angle)

# element_list のフォーマットを fem_lib.py の fem_calc が期待する形式に修正
# [[esm_matrix, start_node_idx, end_node_idx, angle, Ws, We, length]]
# 今回は分布荷重なしなので Ws=0, We=0
element_list = [
    (K_element, 0, 1, angle, 0, 0, L) 
]

# 範囲条件 (境界条件と外力条件)
# ノード0を固定支点、ノード1を自由端とし、ノード1に下向きの集中荷重を適用
node_data = {
    'rc_x': [1, 0], # ノード0: X方向拘束, ノード1: X方向自由
    'rc_y': [1, 0], # ノード0: Y方向拘束, ノード1: Y方向自由
    'rc_m': [1, 0], # ノード0: 回転拘束, ノード1: 回転自由
    'ef_x': [0, 0], # ノード0: X方向外力なし, ノード1: X方向外力なし
    'ef_y': [0, -1000], # ノード0: Y方向外力なし, ノード1: Y方向下向き1000Nの力
    'ef_m': [0, 0]  # ノード0: モーメント外力なし, ノード1: モーメント外力なし
}
nodes_df = pd.DataFrame(node_data)

# 計算実行
result = d_r(element_list, nodes_df)

st.write("### 変位結果")
if result is not None:
    st.write(result)
else:
    st.error("解析に失敗しました。構造が不安定であるか、入力に問題がある可能性があります。")

st.write("---")
st.write("#### 設定パラメータ")
st.write(f"**ヤング率 (E):** {E} N/mm²")
st.write(f"**断面積 (A):** {A} mm²")
st.write(f"**断面二次モーメント (I):** {I} mm⁴")
st.write(f"**部材長さ (L):** {L} mm")
st.write(f"**部材角度 (angle):** {angle} 度")
st.write(f"**ノードデータ:**")
st.dataframe(nodes_df)