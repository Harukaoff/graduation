import sys
import os
sys.path.append(os.path.dirname(__file__))
from fem_lib import esm, d_r
import streamlit as st
import pandas as pd
import numpy as np
from fem_lib import esm, d_r

st.title("小規模 FEM 実行デモ")

# 部材パラメータ
E = 2.0*10**3 
A = 6.0*10**3
I = 2.0*10**7
L = 8000
angle = 0

# 局所剛性行列
K = esm(E, A, I, L, angle)
element_list = [(K, 0, 1,E,A,I,L)]

# 範囲條件
node_data = {
    'rc_x': [1, 0], 'rc_y': [1, 0], 'rc_m': [1, 0],
    'ef_x': [0, 0], 'ef_y': [0, -1000], 'ef_m': [0, 0]
}
nodes_df = pd.DataFrame(node_data)

# 計算
result = d_r(element_list, nodes_df)
st.write("### 変位結果")
st.write(result)