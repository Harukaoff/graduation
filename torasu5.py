import math

def mxmx(a, b, c):
    # c = a @ b
    for i in range(1, 5):
        for j in range(1, 5):
            c[i][j] = 0.0
            for k in range(1, 5):
                c[i][j] += a[i][k] * b[k][j]

def zahyou(t, theta):
    # 4x4座標変換行列
    for i in range(1, 3):
        for j in range(1, 3):
            t[i+2][j] = 0.0
            t[i][j+2] = 0.0
    t[1][1] = math.cos(theta)
    t[1][2] = math.sin(theta)
    t[2][1] = -math.sin(theta)
    t[2][2] = math.cos(theta)
    t[3][3] = t[1][1]
    t[3][4] = t[1][2]
    t[4][3] = t[2][1]
    t[4][4] = t[2][2]

def mxtmx(a, b, c):
    # c = a^T @ b
    for i in range(1, 5):
        for j in range(1, 5):
            c[i][j] = 0.0
            for k in range(1, 5):
                c[i][j] += a[k][i] * b[k][j]

def gausu(n, a, x, b):
    # ガウス消去法による連立一次方程式 a x = b の解
    # a: (n+1)x(n+1), x: (n+1), b: (n+1)  (0番目ダミー)
    for k in range(1, n):
        for i in range(k+1, n+1):
            for j in range(k+1, n+1):
                a[i][j] -= a[k][j] * a[i][k] / a[k][k]
            b[i] -= b[k] * a[i][k] / a[k][k]
    # 後退代入
    x[n] = b[n] / a[n][n]
    for k in range(n-1, 0, -1):
        akjxj = 0.0
        for j in range(k+1, n+1):
            akjxj += a[k][j] * x[j]
        x[k] = (b[k] - akjxj) / a[k][k]

def main():
    # 配列は0番目ダミー
    f = [0.0] * 19 #節点力ベクトル
    d = [0.0] * 19 #節点変位ベクトル
    x = [0.0] * 19 #境界条件（拘束節点に0, その他に1が入る)
    ss = [[0.0]*19 for _ in range(19)] #全体剛性行列
    xv = [0.0] * 10 #節点ごとの変位境界 y方向
    xw = [0.0] * 10 #節点ごとの変位境界 z方向
    fy = [0.0] * 10 #節点ごとの荷重条件 y方向
    fz = [0.0] * 10 #節点ごとの荷重条件 z方向
    hidari = [0] * 10 #各要素の左節点番号
    migi = [0] * 10 #各要素の右節点番号
    s = [[[0.0]*5 for _ in range(5)] for _ in range(10)] #要素剛性行列
    th = [0.0] * 10 #各要素の回転角
    t = [[0.0]*5 for _ in range(5)] #座標変換行列（1要素ごとに計算）
    s44 = [[0.0]*5 for _ in range(5)] #要素剛性行列（1要素ごとに移し替える入れ物）
    ts = [[0.0]*5 for _ in range(5)] #TK（1要素ごとに計算）
    tst = [[0.0]*5 for _ in range(5)] #TKT（1要素ごとに計算）
    sn = [0.0] * 5 #節点力の出力のために追加

    pi = math.pi
    nyou = 5 #要素数
    nset = 4 #節点数

    # 初期化
    for n in range(1, nyou+1):
        for i in range(1, 5):
            for j in range(1, 5):
                s[n][i][j] = 0.0

    for i in range(1, 19):
        d[i] = 0.0
        # f[i] = 0.0
        # x[i] = 0.0
        for j in range(1, 19):
            ss[i][j] = 0.0

    for i in range(1, 10):
        xv[i] = 1.0
        xw[i] = 1.0
        fy[i] = 0.0
        fz[i] = 0.0

    E = 10.0e3 #ヤング率 N/mm^2(MPa)
    ell = 1000.0 #1要素の長さ mm
    A = 10.0**2 #面積 mm^2
    sk1 = E*A/ell #要素1のばね定数
    #要素剛性マトリクス
    #要素1の剛性マトリクス
    s[1][2][2] = sk1; s[1][2][4] = -sk1
    s[1][4][2] = -sk1; s[1][4][4] = sk1

    #要素2から要素4までの剛性マトリクス
    for k in range(2, 5):
        for i in range(1, 5):
            for j in range(1, 5):
                s[k][i][j] = s[1][i][j]

    #要素5の剛性マトリクス
    sk5 = sk1 / math.sqrt(2.0) #要素5のばね定数
    s[5][2][2] = sk5; s[5][2][4] = -sk5
    s[5][4][2] = -sk5; s[5][4][4] = sk5

    #各要素の左節点番号、右節点番号
    hidari[1] = 1; migi[1] = 2
    hidari[2] = 2; migi[2] = 3
    hidari[3] = 3; migi[3] = 4
    hidari[4] = 4; migi[4] = 1
    hidari[5] = 1; migi[5] = 3

    #各要素の回転角（上の左右節点番号がz軸に横たわる状態からの）
    th[1] = 0.0
    th[2] = pi/2.0
    th[3] = pi
    th[4] = pi+pi/2.0
    th[5] = pi/4.0

    # 境界条件
    xv[1] = 0.0; xw[1] = 0.0
    xv[2] = 0.0
    for i in range(1, 10):
        x[2*i-1] = xv[i]
        x[2*i] = xw[i]

    # 荷重条件(y方向:fy[節点番号], z方向:fz[節点番号])
    #fy[2] = 1.0
    fz[3] = 500.0 #[N]
    for i in range(1, 10):
        f[2*i-1] = fy[i]
        f[2*i] = fz[i]

    # 要素ごとのTKTの計算
    for n in range(1, nyou+1):
        zahyou(t, th[n])
        for i in range(1, 5):
            for j in range(1, 5):
                s44[i][j] = s[n][i][j]
        mxtmx(t, s44, ts)
        mxmx(ts, t, tst)
        i1 = 2*hidari[n] - 1
        i2 = 2*hidari[n]
        m1 = 2*migi[n] - 1
        m2 = 2*migi[n]
        ss[i1][i1] += tst[1][1]
        ss[i1][i2] += tst[1][2]
        ss[i1][m1] += tst[1][3]
        ss[i1][m2] += tst[1][4]
        ss[i2][i1] += tst[2][1]
        ss[i2][i2] += tst[2][2]
        ss[i2][m1] += tst[2][3]
        ss[i2][m2] += tst[2][4]
        ss[m1][i1] += tst[3][1]
        ss[m1][i2] += tst[3][2]
        ss[m1][m1] += tst[3][3]
        ss[m1][m2] += tst[3][4]
        ss[m2][i1] += tst[4][1]
        ss[m2][i2] += tst[4][2]
        ss[m2][m1] += tst[4][3]
        ss[m2][m2] += tst[4][4]

    # 境界条件を入れる
    for i in range(1, nset*2+1):
        for j in range(1, nset*2+1):
            ss[i][j] = x[i]*ss[i][j]
            ss[j][i] = x[i]*ss[j][i]
    for i in range(1, nset*2+1):
        if x[i] < 1e-3:
            ss[i][i] = 1.0

    gausu(nset*2, ss, d, f)

    for n in range(1, nset+1):
        print(f"節点番号：{n} v={d[n*2-1]:.10f},  w={d[n*2]:.10f}")

    # 要素ごとのTKTの計算
    for n in range(1, nyou+1):
        zahyou(t, th[n])
        for i in range(1, 5):
            for j in range(1, 5):
                s44[i][j] = s[n][i][j]
        mxtmx(t, s44, ts)
        mxmx(ts, t, tst)
        for i in range(1, 5):
            sn[i] = 0.0
            for j in range(1, 3):
                sn[i] += tst[i][j] * d[(hidari[n]-1)*2 + j]
            for j in range(3, 5):
                sn[i] += tst[i][j] * d[(migi[n]-1)*2 + j-2]
        print(f"部材番号： {n}  S({hidari[n]})={sn[1]:15.10f},  N({hidari[n]})={sn[2]:15.10f}")
        print(f"              S({migi[n]})={sn[3]:15.10f},  N({migi[n]})={sn[4]:15.10f}")

if __name__ == "__main__":
    main()

