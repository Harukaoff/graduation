import numpy as np
np.set_printoptions(precision=4)
import pandas as pd
pd.set_option('display.max_rows', None)
import math
import warnings
warnings.simplefilter('ignore')
import draw_lib  # 描画ライブラリ（別途用意）

# ==========================================================
# 変換マトリクス
# ==========================================================
def make_T3(angle):
    mu = math.sin(math.radians(angle))
    lamb = math.cos(math.radians(angle))

    T3 = np.array([
        [ lamb,   mu, 0,    0,    0, 0],
        [  -mu, lamb, 0,    0,    0, 0],
        [    0,    0, 1,    0,    0, 0],
        [    0,    0, 0, lamb,   mu, 0],
        [    0,    0, 0,  -mu, lamb, 0],
        [    0,    0, 0,    0,    0, 1]
    ])
    return T3


# ==========================================================
# 要素剛性マトリクス
# ==========================================================
def esm(E, A, I, L, angle):
    matrix_L = np.array([
        [ (E*A)/L,              0,             0, -(E*A)/L,              0,             0],
        [       0,  (12*E*I)/L**3,  (6*E*I)/L**2,        0, -(12*E*I)/L**3,  (6*E*I)/L**2],
        [       0,   (6*E*I)/L**2,     (4*E*I)/L,        0,  -(6*E*I)/L**2,     (2*E*I)/L],
        [-(E*A)/L,              0,             0,  (E*A)/L,              0,             0],
        [       0, -(12*E*I)/L**3, -(6*E*I)/L**2,        0,  (12*E*I)/L**3, -(6*E*I)/L**2],
        [       0,   (6*E*I)/L**2,     (2*E*I)/L,        0,  -(6*E*I)/L**2,     (4*E*I)/L]
    ])
    matrix_T3 = make_T3(angle)
    matrix_G = np.dot(matrix_T3.T, np.dot(matrix_L, matrix_T3))
    return matrix_G


# ==========================================================
# 全体剛性マトリクス作成
# ==========================================================
def gsm(matrixs):
    node = max(max([i[1] for i in matrixs])+1, max([i[2] for i in matrixs])+1)
    matrix = np.zeros((node*3, node*3))

    for i in matrixs:
        arr = np.zeros((node*3, node*3))
        arr[i[1]*3:i[1]*3+3, i[1]*3:i[1]*3+3] = i[0][:3, :3]
        arr[i[1]*3:i[1]*3+3, i[2]*3:i[2]*3+3] = i[0][:3, 3:]
        arr[i[2]*3:i[2]*3+3, i[1]*3:i[1]*3+3] = i[0][3:, :3]
        arr[i[2]*3:i[2]*3+3, i[2]*3:i[2]*3+3] = i[0][3:, 3:]
        matrix += arr

    return matrix


# ==========================================================
# 分布荷重考慮ベクトル作成
# ==========================================================
def d_l(e_l):
    qm_l = []
    for i in e_l:
        length, angle, Ws, We, start, end = i[6], i[3], i[4], i[5], i[1], i[2]
        x = round(math.sin(math.radians(angle)),3)
        y = round(math.cos(math.radians(angle)),3)

        if 0 <= angle < 90:
            x = -x
        elif 90 <= angle < 180:
            x, y = -x, -y
        elif 180 <= angle < 270:
            y = -y
        elif 270 <= angle:
            x = -x

        Qxs = (length/20)*(7*Ws*x + 3*We*x)
        Qxe = (length/20)*(3*Ws*x + 7*We*x)
        Qys = (length/20)*(7*Ws*y + 3*We*y)
        Qye = (length/20)*(3*Ws*y + 7*We*y)
        Ms = (length**2/60)*(3*Ws + 2*We)
        Me = -(length**2/60)*(2*Ws + 3*We)

        qm_l += [[start, [Qxs, Qys, Ms]], [end, [Qxe, Qye, Me]]]

    dl_dict = {}
    for index, values in qm_l:
        dl_dict[index] = [sum(x) for x in zip(dl_dict.get(index, [0,0,0]), values)]
    return [v for k, v in sorted(dl_dict.items()) for v in v]


# ==========================================================
# 各接点の変位・反力計算
# ==========================================================
def d_r(e_l, n_d):
    matrix = gsm(e_l)
    dl = d_l(e_l)
    rc = n_d[['rc_x','rc_y','rc_m']].values.tolist()
    ef = n_d[['ef_x','ef_y','ef_m']].values.tolist()

    matrix_ind = matrix.shape[0]
    rc_ind = [3*r+c for r,row in enumerate(rc) for c,v in enumerate(row) if v==1]
    ef_ind = [i for i in range(matrix_ind) if i not in rc_ind]

    aa = [dl[index] for index in ef_ind]
    ba = [dl[index] for index in rc_ind]

    mat = matrix[:, [i not in rc_ind for i in range(matrix_ind)]]
    Kaa = mat[[i not in rc_ind for i in range(matrix_ind)]]
    Kba = mat[[i in rc_ind for i in range(matrix_ind)]]

    Pa = [sum(ef, [])[i]-aa[j] for j,i in enumerate(ef_ind)]
    Ua = np.linalg.pinv(Kaa) @ Pa
    Pb = (Kba @ Ua) + ba

    Pa_list = [[x,y] for x,y in zip(ef_ind,Pa)]
    Pb_list = [[x,y] for x,y in zip(rc_ind,Pb)]
    Ua_list = [[x,y] for x,y in zip(ef_ind,Ua)]

    df = pd.DataFrame(index=range(int(matrix_ind/3)),
                      columns=['Px','Py','M','u','v','theta']).fillna(0)

    for i in Pa_list:
        ind,col = divmod(i[0],3)
        df.iat[ind,col] = i[1]+dl[i[0]]
    for i in Pb_list:
        ind,col = divmod(i[0],3)
        df.iat[ind,col] = i[1]
    for i in Ua_list:
        ind,col = divmod(i[0],3)
        df.iat[ind,col+3] = i[1]
    return df


# ==========================================================
# 部材応力＋変形計算（詳細）
# ==========================================================
def member_stress(e_l, d_r, n_d):
    step = 5
    node = n_d[['x','y']].values.tolist()
    disp_df_list = []

    for i in e_l:
        length, angle, Ws, We = i[6], i[3], i[4], i[5]
        sin, cos = math.sin(math.radians(angle)), math.cos(math.radians(angle))
        x = round(math.sin(math.radians(angle)),3)
        y = round(math.cos(math.radians(angle)),3)
        if 0 <= angle < 90:
            x = -x
        elif 90 <= angle < 180:
            x, y = -x, -y
        elif 180 <= angle < 270:
            y = -y
        elif 270 <= angle:
            x = -x

        Qxs = (length/20)*(7*Ws*x + 3*We*x)
        Qxe = (length/20)*(3*Ws*x + 7*We*x)
        Qys = (length/20)*(7*Ws*y + 3*We*y)
        Qye = (length/20)*(3*Ws*y + 7*We*y)
        Ms = (length**2/60)*(3*Ws + 2*We)
        Me = -(length**2/60)*(2*Ws + 3*We)

        if Ws*x!=0 or Ws*y!=0 or We*x!=0 or We*y!=0:
            Wxy = [Ws*x, -Ws*y, 1, We*x, -We*y, 1]
        else:
            Wxy = [0,0,0,0,0,0]

        Kg = i[0]
        Ug = d_r.iloc[i[1],3:].values.tolist()+d_r.iloc[i[2],3:].values.tolist()
        Fg_G = (Kg @ Ug)+[Qxs,Qys,Ms,Qxe,Qye,Me]
        T3 = make_T3(i[3])
        Fg_L = (T3 @ Fg_G).tolist()
        Fg_L = [-Fg_L[0], Fg_L[1], Fg_L[2], Fg_L[3], -Fg_L[4], -Fg_L[5]]
        Wuv = T3 @ Wxy

        start = node[i[1]]
        d_r_list = d_r.iloc[[i[1], i[2]],3:].values.tolist()
        Us,Vs,Ts = d_r_list[0]
        Ue,Ve,Te = d_r_list[1]
        Us_l,Vs_l = Us*cos + Vs*sin, -Us*sin + Vs*cos
        Ue_l,Ve_l = Ue*cos + Ve*sin, -Ue*sin + Ve*cos

        disp_list = []
        for x_ in range(0, int(length)+1, step):
            Ux = (1-x_/length)*Us_l + (x_/length)*Ue_l
            Vx = np.array([
                (1-3*x_**2/length**2+2*x_**3/length**3),
                (x_-2*x_**2/length+x_**3/length**2),
                (3*x_**2/length**2-2*x_**3/length**3),
                (-x_**2/length+x_**3/length**2)
            ]) @ [Vs_l, Ts, Ve_l, Te]
            dx = Ux*cos - Vx*sin
            dy = Ux*sin + Vx*cos
            c_x = x_*cos - 0*sin + start[0]
            c_y = x_*sin + 0*cos + start[1]
            N = -(Wuv[3]-Wuv[0])/2*x_**2/length - Wuv[0]*x_ + Fg_L[0]
            Q = (Wuv[4]-Wuv[1])/2*x_**2/length + Wuv[1]*x_ + Fg_L[1]
            M = (Wuv[4]-Wuv[1])/6*x_**3/length + Wuv[1]/2*x_**2 + Fg_L[1]*x_ - Fg_L[2]
            disp_list.append([x_, c_x, c_y, dx, dy, N, Q, M])

        disp_df = pd.DataFrame(disp_list, columns=['delta','x','y','dx','dy','N','Q','M'])
        if len(disp_df[(disp_df['x']==node[i[2]][0]) & (disp_df['y']==node[i[2]][1])]) == 0:
            disp_df.iloc[-1] = [length, node[i[2]][0], node[i[2]][1], Ue, Ve, Fg_L[3], Fg_L[4], -Fg_L[5]]
        disp_df_list.append(disp_df)
    return disp_df_list


# ==========================================================
# 構造全体の形状関数（変形図用）
# ==========================================================
def shape_func(e_l, d_r, n_d):
    step = 1
    mag = 15
    node = n_d[['x','y']].values.tolist()

    dr_list=[]
    for i in e_l:
        length,angle=i[6],i[3]
        sin,cos=math.sin(math.radians(angle)),math.cos(math.radians(angle))
        start=node[i[1]]
        d_r_list=d_r.iloc[[i[1],i[2]],3:].values.tolist()
        Us,Vs,Ts=d_r_list[0]
        Ue,Ve,Te=d_r_list[1]
        Us_l,Vs_l=Us*cos+Vs*sin,-Us*sin+Vs*cos
        Ue_l,Ve_l=Ue*cos+Ve*sin,-Ue*sin+Ve*cos
        for x in range(0,int(length)+1,step):
            Ux=(1-x/length)*Us_l+(x/length)*Ue_l
            Vx=np.array([
                (1-3*x**2/length**2+2*x**3/length**3),
                (x-2*x**2/length+x**3/length**2),
                (3*x**2/length**2-2*x**3/length**3),
                (-x**2/length+x**3/length**2)
            ])@[Vs_l,Ts,Ve_l,Te]
            dr_list.append(Ux)
            dr_list.append(Vx)

    max_dr=max(map(abs,dr_list))

    disp_df_list=[]
    for i in e_l:
        length,angle=i[6],i[3]
        sin,cos=math.sin(math.radians(angle)),math.cos(math.radians(angle))
        start=node[i[1]]
        d_r_list=d_r.iloc[[i[1],i[2]],3:].values.tolist()
        Us,Vs,Ts=d_r_list[0]
        Ue,Ve,Te=d_r_list[1]
        Us_l,Vs_l=Us*cos+Vs*sin,-Us*sin+Vs*cos
        Ue_l,Ve_l=Ue*cos+Ve*sin,-Ue*sin+Ve*cos

        disp_list=[]
        for x in range(0,int(length)+1,step):
            Ux=(1-x/length)*Us_l+(x/length)*Ue_l
            Vx=np.array([
                (1-3*x**2/length**2+2*x**3/length**3),
                (x-2*x**2/length+x**3/length**2),
                (3*x**2/length**2-2*x**3/length**3),
                (-x**2/length+x**3/length**2)
            ])@[Vs_l,Ts,Ve_l,Te]
            X_l=start[0]+x+(Ux/max_dr)*mag
            Y_l=start[1]+(Vx/max_dr)*mag
            X_g=(X_l-start[0])*cos-(Y_l-start[1])*sin+start[0]
            Y_g=(X_l-start[0])*sin+(Y_l-start[1])*cos+start[1]
            disp_list.append([x,X_g,Y_g])
        disp_df=pd.DataFrame(disp_list,columns=['x','X_g','Y_g'])
        disp_df_list.append(disp_df)
    return disp_df_list
