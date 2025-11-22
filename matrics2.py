import numpy as np

# パラメータ
E = 2e7
A = 0.01
L_elem = 1.0
k_elem = (E * A) / L_elem

# 要素剛性行列（共通）
k_local = k_elem * np.array([[1, -1], [-1, 1]])

# 全体剛性行列（3節点 → 3x3）
K = np.zeros((3, 3))

# 要素1（節点0-1）をアセンブル
K[0:2, 0:2] += k_local

# 要素2（節点1-2）をアセンブル
K[1:3, 1:3] += k_local

print("全体剛性行列 K：\n", K)

import numpy as np

# 全体剛性行列（前ステップのもの）
K = np.array([
    [200000, -200000, 0],
    [-200000, 400000, -200000],
    [0, -200000, 200000]
], dtype=np.float64)

# 荷重ベクトル（節点2に1000Nの外力）
F = np.array([0, 0, 1000], dtype=np.float64)

# 変位ベクトル（未知の u1, u2 を解きたい）
u = np.zeros(3)

# 節点0を固定 → u[0] = 0
# 縮小剛性行列・荷重ベクトルを作成
K_reduced = K[1:, 1:]
F_reduced = F[1:]

# 変位を解く
u_reduced = np.linalg.solve(K_reduced, F_reduced)

# 結果を代入
u[0] = 0  # 固定
u[1:] = u_reduced

print("各節点の変位 u = ", u)
