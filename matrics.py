import numpy as np

# 条件
A = 0.01        # m²
E = 200e9       # Pa
L = 2.0         # m
k = A * E / L

# 剛性マトリクス
K = k * np.array([
    [1, -1],
    [-1, 1]
])

# 外力ベクトル（節点1=固定、節点2に1000N）
F = np.array([0, 1000])

# 節点1が固定 → u1 = 0
# 式の2行目だけ抜き出して解く
# -k * u1 + k * u2 = 1000 → k * u2 = 1000
u2 = F[1] / k
u1 = 0

# 全体の変位ベクトル
u = np.array([u1, u2])
print("変位 u =", u)

# 軸力の計算
delta_u = u[1] - u[0]
N = E * A * delta_u / L

# 応力の計算
stress = N / A

print(f"内部軸力 N = {N:.2f} N")
print(f"応力 σ = {stress:.2f} Pa")
