import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fem_lib import esm, d_r

st.title("構造解析＋モーメント図出力（最小モデル）")

# 材料特性と部材寸法（画像から得る値を仮で入力）
E = 210000.0
A = 300.0
I = 200.0
L = 4000.0
angle = 0.0

# 剛性マトリクス作成（画像から接続点0→1と仮定）
K = esm(E, A, I, L, angle)
element_list = [(K, 0, 1, E, A, I, L)]

# 支点と荷重条件（Node 0 固定、Node 1 に下向き荷重）
node_data = {
    'rc_x': [1, 0], 'rc_y': [1, 0], 'rc_m': [1, 0],
    'ef_x': [0, 0], 'ef_y': [0, -1000], 'ef_m': [0, 0]
}
nodes_df = pd.DataFrame(node_data)

# 変位と反力の計算
result_df = d_r(element_list, nodes_df)
st.write(result_df)

# モーメント図を描画（要素内線形分布と仮定）
def plot_moment(E, I, L, displacement):
    x = np.linspace(0, L, 100)
    # 単純な近似式（Node 1に荷重がある仮定）
    M = -1000 * (L - x)  # 単純梁 + 荷重右端パターン
    fig, ax = plt.subplots()
    ax.plot(x, M, label="モーメント図")
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel("位置 x [mm]")
    ax.set_ylabel("曲げモーメント M [N·mm]")
    ax.legend()
    st.pyplot(fig)

st.subheader("モーメント図")
plot_moment(E, I, L, result_df)
