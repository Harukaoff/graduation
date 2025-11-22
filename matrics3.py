import numpy as np

# 材料定数と部材長さ
E = 200e9  # ヤング率 [Pa]
A = 0.01   # 断面積 [m^2]
L = 1.0    # 各要素の長さ [m]

# 節点の変位 [m]（前ステップの結果）
u = np.array([0.0, 0.005, 0.01])

# 各要素の軸力を求める関数
def axial_force(u_i, u_j, E, A, L):
    return (E * A / L) * (u_j - u_i)

# 要素1（節点0-1）
N1 = axial_force(u[0], u[1], E, A, L)
# 要素2（節点1-2）
N2 = axial_force(u[1], u[2], E, A, L)

print(f"要素1の軸力: {N1:.2f} N")  # 引張なら正、圧縮なら負
print(f"要素2の軸力: {N2:.2f} N")
