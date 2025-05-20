# -*- coding: utf-8 -*-
# Cording by Sogo Iwao from nasg25(m42) on 2024-5-22.

from merge_airfoil import Merge_Airfoil
from airfoil_thickness import AirfoilThickness

from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
import pandas as pd
from pathlib import Path
import re
import time
import numpy as np
import random
import matplotlib.pyplot as plt


class Read:
    def __init__(self, dir_1, alpha):
        self.dir = "airfoil_data\\" + dir_1
        self.alpha = alpha
        self.Re_delta = 5000

    def read_data(self):
        print('---Read_start---')
        start = time.time()
        path = Path.cwd()
        airfoil_data = path / self.dir
        airfoil_data_g = airfoil_data.iterdir()
        paths = [str(z) for z in airfoil_data_g]

        for z1 in range(len(paths)):
            df_1 = pd.read_csv(paths[z1], usecols=[0, 1, 2, 4], header=None, skiprows=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                               names=['alpha', 'Cl', 'Cd', 'Cm'], engine='python')

            m = re.search(r'Re0.[0-9]{3}', paths[z1])
            if m:
                Re_str = m.group()
            else:
                Re_str = None

            m1 = re.search(r'_[0-9]*_', paths[z1])
            if m1:
                mix_str = m1.group()
            else:
                mix_str = None
            Re = int(re.sub(r'\D', '', Re_str))
            mix = int(re.sub(r'\D', '', mix_str))
            df_1['Re'] = Re * 1000
            df_1['mix'] = mix

            if z1 == 0:
                self.df = df_1
            else:
                self.df = pd.concat([self.df, df_1], ignore_index=True)

        self.Re_max = self.df['Re'].max()
        self.Re_min = self.df['Re'].min()
        #print(self.df)
        end = time.time()
        print('処理時間:', end - start)
        print('---Read_end---')
        return self.df, self.Re_max, self.Re_min

    def interpolate_cd(self):
        df_alpha = self.df[self.df['alpha'] == self.alpha]
        df_alpha_sort = df_alpha.sort_values(['Re', 'mix'])
        df_alpha_sort['Re*Cd'] = df_alpha_sort['Re'] * df_alpha_sort['Cd']
        # df_alpha_sort['Re*Cl'] = df_alpha_sort['Re'] * df_alpha_sort['Cl']
        df_alpha_sort_Re_Cd = df_alpha_sort[['Re', 'mix', 'Re*Cd']]
        df_alpha_sort_Re_Cd_drop_index = df_alpha_sort_Re_Cd.index[df_alpha_sort_Re_Cd['Re'] % self.Re_delta != 0]
        df_alpha_sort_Re_Cd_bug_bug = df_alpha_sort_Re_Cd.drop(df_alpha_sort_Re_Cd_drop_index)
        df_alpha_sort_Re_Cd_re_bug = df_alpha_sort_Re_Cd_bug_bug.set_index(['Re', 'mix'])
        df_alpha_sort_Re_Cd_re = df_alpha_sort_Re_Cd_re_bug.drop_duplicates()
        df_alpha_sort_Re_Cd_unstack = df_alpha_sort_Re_Cd_re.unstack()
        # print(df_alpha_sort_Re_Cd_unstack )
        df_alpha_sort_Re_Cd_unstack_interpolate = df_alpha_sort_Re_Cd_unstack.interpolate()

        foil_alpha_Re_Cd = df_alpha_sort_Re_Cd_unstack_interpolate.to_numpy()
        np.set_printoptions(threshold=1000)
        # print(np.get_printoptions())
        print(df_alpha_sort_Re_Cd_unstack)
        # 行の表示を省略しない
        # pd.set_option('display.max_rows', 100)

        # 列の表示を省略しない
        # pd.set_option('display.max_columns', none)

        # print(df_alpha_sort_Re_Cd_unstack_interpolate )
        print(foil_alpha_Re_Cd)
        # 解析データのRe数の幅と間隔
        x = range(self.Re_min, self.Re_max + 1, self.Re_delta)
        # 解析データの翼型混合率の幅と間隔
        y = range(0, 101, 5)
        print('---interpolate_cd_start---')
        start1 = time.time()
        self.fx_cd_2d = RegularGridInterpolator((x, y), foil_alpha_Re_Cd, method='cubic')
        end1 = time.time()
        print('---interpolate_cd_end---')
        print(start1 - end1)
        return self.fx_cd_2d

    def interpolate_gamma(self):
        df_alpha_2 = self.df[self.df['alpha'] == self.alpha]
        df_alpha_sort_2 = df_alpha_2.sort_values(['Re', 'mix'])
        # df_alpha_sort['Re*Cd'] = df_alpha_sort['Re'] * df_alpha_sort['Cd']
        df_alpha_sort_2['Re*Cl'] = df_alpha_sort_2['Re'] * df_alpha_sort_2['Cl']
        df_alpha_sort_Re_Cl = df_alpha_sort_2[['Re', 'mix', 'Re*Cl']]
        df_alpha_sort_Re_Cl_drop_index = df_alpha_sort_Re_Cl.index[df_alpha_sort_Re_Cl['Re'] % self.Re_delta != 0]
        df_alpha_sort_Re_Cl_bug_bug = df_alpha_sort_Re_Cl.drop(df_alpha_sort_Re_Cl_drop_index)
        df_alpha_sort_Re_Cl_re_bug = df_alpha_sort_Re_Cl_bug_bug.set_index(['Re', 'mix'])
        df_alpha_sort_Re_Cl_re = df_alpha_sort_Re_Cl_re_bug.drop_duplicates()
        df_alpha_sort_Re_Cl_unstack = df_alpha_sort_Re_Cl_re.unstack()
        # print(df_alpha_sort_Re_Cl_unstack )
        df_alpha_sort_Re_Cl_unstack_interpolate = df_alpha_sort_Re_Cl_unstack.interpolate()
        foil_alpha_Re_Cl = df_alpha_sort_Re_Cl_unstack_interpolate.to_numpy()

        # 解析データのRe数の幅と間隔
        x = range(self.Re_min, self.Re_max + 1, self.Re_delta)
        # 解析データの翼型混合率の幅と間隔
        y = range(0, 101, 5)

        print('---interpolate_cl_start---')
        start2 = time.time()
        self.fx_cl_2d = RegularGridInterpolator((x, y), foil_alpha_Re_Cl, method='cubic')
        end2 = time.time()
        print('---interpolate_cl_end---')
        print(start2 - end2)
        return self.fx_cl_2d


class PSO:
    def __init__(self, dir2, U, alpha, nu, gamma_object, thickness_min_list, aifolil_root, airfoil_tip, spar_loc):
        self.dir = dir2
        self.alpha = alpha
        self.U = U
        self.nu = nu
        self.gamma_list = gamma_object
        self.thickness_min_list = np.array(thickness_min_list) / 1000 # mm to m
        self.airfoil_root = aifolil_root
        self.airfoil_tip = airfoil_tip
        self.x1 = AirfoilThickness(self.airfoil_root, spar_loc)
        self.max_thickness_R, self.max_thickness_loc_R, self.thickness_loc_R = self.x1.airfoil_thickness()
        self.x2 = AirfoilThickness(self.airfoil_tip, spar_loc)
        self.max_thickness_T, self.max_thickness_loc_T, self.thickness_loc_T = self.x2.airfoil_thickness()
        self.max_thickness_gradient = (self.max_thickness_T - self.max_thickness_R) / 100
        self.thickness_loc_gradient = (self.thickness_loc_T - self.thickness_loc_R) / 100
        if spar_loc == None:
            self.thickness_gradient = self.max_thickness_gradient
            self.thickness_R = self.max_thickness_R
        else:
            self.thickness_gradient = self.thickness_loc_gradient
            self.thickness_R = self.thickness_loc_R
        read = Read(self.dir, self.alpha)
        self.df, self.Re_max, self.Re_min = read.read_data()
        self.fx_cd = read.interpolate_cd()
        self.fx_cl = read.interpolate_gamma()
        self.chord_list = []
        self.mix_list = []
        self.drag_list = []
        self.gamma_zure_list = []
        self.thickness_min = []

    # def read_fx(self):
    # read = Read(self.dir,self.alpha)
    # read.read_data()
    # self.fx_cd = read.interpolate_cd()
    # self.fx_cl = read.interpolate_gamma()

    def fitness(self, Re, m):
        # print(Re)
        Re = np.clip(Re, self.Re_min, self.Re_max)
        m = np.clip(m, 0, 100)
        z = self.fx_cd((Re, m))
        return z

    def constraints(self, Re, m):
        Re = np.clip(Re, self.Re_min, self.Re_max)
        m = np.clip(m, 0, 100)
        gamma = self.fx_cl((Re, m)) * self.nu / 2
        mu_gamma = abs(1 - gamma / self.gamma_object)

        # m（混合率）から線形補間で最大翼厚を計算
        max_thickness = self.thickness_R + (self.thickness_gradient * m)
        thickness = max_thickness * Re / self.U * self.nu
        mu_thickness = 0
        if self.thickness_min is not None and thickness < self.thickness_min:
            mu_thickness = abs(self.thickness_min - thickness)

        return mu_gamma + mu_thickness * 10

    def update_position(self, x1, x2, vx1, vx2):
        new_x1 = x1 + vx1
        new_x2 = x2 + vx2
        if new_x1 > self.Re_max:
            new_x1 = self.Re_max
        elif new_x1 < 30000:
            new_x1 = 30000
        elif new_x2 > 100:
            new_x2 = 100
        elif new_x2 < 0:
            new_x2 = 0
        return new_x1, new_x2

    def update_velocity(self, x1, x2, vx1, vx2, p, g, T, t, rho_max=0.14):
        # パラメーターrhoはランダムに与える
        rho1 = random.uniform(0, rho_max)
        rho2 = random.uniform(0, rho_max)
        # 慣性項の変数ｗを変える
        if self.gamma_object >= 2.8:
            w_start = 0.9
            w_end = 0.2
            w_gradient = (w_end - w_start) / T
            w = w_gradient * t + w_start
            # 粒子速度の更新を行う
            new_vx1 = w * vx1 + rho1 * (p["x1"] - x1) + rho2 * (g["x1"] - x1)
            new_vx2 = w * vx2 + rho1 * (p["x2"] - x2) + rho2 * (g["x2"] - x2)
            return new_vx1, new_vx2
        else:
            w_start = 0.7
            w_end = 0.1
            w_gradient = (w_end - w_start) / T
            w = w_gradient * t + w_start
            # 粒子速度の更新を行う
            new_vx1 = w * vx1 + rho1 * (p["x1"] - x1) + rho2 * (g["x1"] - x1)
            new_vx2 = w * vx2 + rho1 * (p["x2"] - x2) + rho2 * (g["x2"] - x2)
            return new_vx1, new_vx2

    def pso(self):
        # print('---PSO_start---')
        N = 1000  # 粒子の数
        x1_min, x1_max = 30000, self.Re_max
        x2_min, x2_max = 0, 100

        # 粒子位置, 速度, パーソナルベスト, グローバルベストの初期化を行う
        ps = [{"x1": random.uniform(x1_min, x1_max), "x2": random.uniform(x2_min, x2_max)} for i in range(N)]
        vs = [{"x1": 0.0, "x2": 0.0} for i in range(N)]
        personal_best_positions = ps
        personal_best_scores = [x.fitness(p["x1"], p["x2"]) for p in ps]
        personal_mu = [x.constraints(a["x1"], a["x2"]) for a in ps]
        fx_pass = []
        fx_pass_index = []
        mu_ave = sum(personal_mu) / N
        mu_min = min(personal_mu)
        alpha_level_initial = (mu_ave + mu_min) / 2
        for i in range(N):
            if alpha_level_initial > personal_mu[i]:
                fx_pass.append(personal_best_scores[i])
                fx_pass_index.append(i)

        best_particle = np.argmin(fx_pass)
        global_best_position = personal_best_positions[fx_pass_index[best_particle]]
        # print(mu_min)
        # print(alpha_level_initial)
        # print(personal_best_positions)
        # print(personal_best_scores)
        # print(best_particle)
        # print(global_best_position)
        #print('---PSO_start---')
        T = 30  # 制限時間(ループの回数)
        for t in range(T):
            fx_pass.clear()
            fx_pass_index.clear()
            if t < T / 2:

                alpha_level = alpha_level_initial * (1 - 2 * t / T) ** 2
                # print(alpha_level)
                for i in range(N):
                    x1, x2 = ps[i]["x1"], ps[i]["x2"]
                    vx1, vx2 = vs[i]["x1"], vs[i]["x2"]
                    p = personal_best_positions[i]
                    # 粒子の位置の更新を行う
                    new_x1, new_x2 = x.update_position(x1, x2, vx1, vx2)
                    ps[i] = {"x1": new_x1, "x2": new_x2}
                    # 粒子の速度の更新を行う
                    new_vx1, new_vx2 = x.update_velocity(new_x1, new_x2, vx1, vx2, p, global_best_position, T, t)
                    vs[i] = {"x1": new_vx1, "x2": new_vx2}
                    # 評価値を求め, パーソナルベストの更新を行う
                    score = x.fitness(new_x1, new_x2)
                    mu_score = x.constraints(new_x1, new_x2)
                    if mu_score < alpha_level and personal_mu[i] < alpha_level:
                        if score < personal_best_scores[i]:
                            personal_best_scores[i] = score
                            personal_mu[i] = mu_score
                            personal_best_positions[i] = {"x1": new_x1, "x2": new_x2}

                    elif mu_score == personal_mu[i]:
                        personal_best_scores[i] = score
                        personal_mu[i] = mu_score
                        personal_best_positions[i] = {"x1": new_x1, "x2": new_x2}

                    else:
                        if mu_score < personal_mu[i]:
                            personal_best_scores[i] = score
                            personal_mu[i] = mu_score
                            personal_best_positions[i] = {"x1": new_x1, "x2": new_x2}

                # グローバルベストの更新を行う
                for i in range(N):
                    if alpha_level > personal_mu[i]:
                        fx_pass.append(personal_best_scores[i])
                        fx_pass_index.append(i)

                if len(fx_pass) == 0:
                    best_particle = np.argmin(personal_mu)
                    global_best_position = personal_best_positions[best_particle]
                else:
                    best_particle = np.argmin(fx_pass)
                    global_best_position = personal_best_positions[fx_pass_index[best_particle]]

                # print(len(fx_pass))

            else:

                alpha_level = 0
                for i in range(N):
                    x1, x2 = ps[i]["x1"], ps[i]["x2"]
                    vx1, vx2 = vs[i]["x1"], vs[i]["x2"]
                    p = personal_best_positions[i]
                    # 粒子の位置の更新を行う
                    new_x1, new_x2 = x.update_position(x1, x2, vx1, vx2)
                    ps[i] = {"x1": new_x1, "x2": new_x2}
                    # 粒子の速度の更新を行う
                    new_vx1, new_vx2 = x.update_velocity(new_x1, new_x2, vx1, vx2, p, global_best_position, T, t)
                    vs[i] = {"x1": new_vx1, "x2": new_vx2}
                    # 評価値を求め, パーソナルベストの更新を行う
                    score = x.fitness(new_x1, new_x2)
                    mu_score = x.constraints(new_x1, new_x2)
                    if mu_score < alpha_level and personal_mu[i] < alpha_level:
                        if score < personal_best_scores[i]:
                            personal_best_scores[i] = score
                            personal_mu[i] = mu_score
                            personal_best_positions[i] = {"x1": new_x1, "x2": new_x2}

                    elif mu_score == personal_mu[i]:
                        personal_best_scores[i] = score
                        personal_mu[i] = mu_score
                        personal_best_positions[i] = {"x1": new_x1, "x2": new_x2}

                    else:
                        if mu_score < personal_mu[i]:
                            personal_best_scores[i] = score
                            personal_mu[i] = mu_score
                            personal_best_positions[i] = {"x1": new_x1, "x2": new_x2}

                # グローバルベストの更新を行う
                for i in range(N):
                    if alpha_level > personal_mu[i]:
                        fx_pass.append(personal_best_scores[i])
                        fx_pass_index.append(i)

                if len(fx_pass) == 0:
                    best_particle = np.argmin(personal_mu)
                    global_best_position = personal_best_positions[best_particle]
                else:
                    best_particle = np.argmin(fx_pass)
                    global_best_position = personal_best_positions[fx_pass_index[best_particle]]

        print('---PSO_end---')
        print('最適粒子位置(Re,mix):', global_best_position)
        print('最小評価値:', np.min(personal_best_scores))
        print('制約違反度:', x.constraints(global_best_position["x1"], global_best_position["x2"]))
        print('最適コード長:', global_best_position["x1"] * self.nu / self.U)
        print('翼型厚さ:', global_best_position["x1"] * self.nu / self.U*(self.thickness_R + (self.thickness_gradient * global_best_position["x2"])))

        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(projection='3d')
        # x1_coord = np.linspace(x1_min, x1_max, 100)
        # x2_coord = np.linspace(x2_min, x2_max, 100)
        # X, Y = np.meshgrid(x1_coord, x2_coord)
        # ax1.plot_wireframe(X, Y, x.fitness(X, Y), color='b', rstride=2, cstride=2, linewidth=0.3)
        # ax1.set_xlabel('$Re$')
        # ax1.set_ylabel('$m$')
        # ax1.set_zlabel('$Re*Cd$')
        # ax1.scatter3D(global_best_position['x1'], global_best_position['x2'], np.min(personal_best_scores), color='red')

        return (global_best_position['x1'] * self.nu / self.U, global_best_position['x2'],
                x.constraints(global_best_position["x1"], global_best_position["x2"]), np.min(personal_best_scores),
                global_best_position["x1"] * self.nu / self.U * (self.thickness_R + (self.thickness_gradient * global_best_position["x2"]))*1000)

    def execute(self):
        for i1 in range(len(self.gamma_list)):
            zure = 1
            output = None
            while zure > 0.01:
                self.gamma_object = self.gamma_list[i1]
                self.thickness_min = self.thickness_min_list[i1]
                print('\n' + '='*40)
                print(f'🚀🚀🚀   PSO アルゴリズム開始   🚀🚀🚀   ＜{i1+1}区間目＞')
                print('='*40 + '\n')
                output = self.pso()
                zure = output[2]
            self.chord_list.append(output[0])
            self.mix_list.append(output[1])
            self.gamma_zure_list.append(output[2])
            self.drag_list.append(output[3])
            self.thickness_min_list[i1] = output[4]
        return self.chord_list, self.mix_list, self.gamma_zure_list, self.drag_list, self.thickness_min_list

class Calcurate_Cd(PSO):
    def __init__(self, break_point, chord_list, mix_list, dir2, U, alpha, nu, gamma_object, thickness_min_list, airfoil_root, airfoil_tip, spar_loc):
        super().__init__(dir2, U, alpha, nu, gamma_object, thickness_min_list, airfoil_root, airfoil_tip, spar_loc)
        self.break_point = break_point
        self.chord_list = chord_list
        self.mix_list = mix_list
        self.chord_fx = interpolate.interp1d(break_point, chord_list)
        self.mix_fx = interpolate.interp1d(break_point, mix_list)

    def calcularate_cd(self):
        n_cd = 50
        mu = 1.869e-5
        D_n = 0
        for i in range(len(self.break_point) - 1):
            brock_span = self.break_point[i + 1] - self.break_point[i]
            delta_c = brock_span / n_cd
            for j in range(n_cd):
                y = self.break_point[i] + delta_c * (0.5 + j)
                Re_n = self.chord_fx(y) * self.U / self.nu
                mix_n = self.mix_fx(y)
                Re_Cd_n = self.fx_cd((Re_n, mix_n))
                D_n += mu * self.U * delta_c * Re_Cd_n / 2

        return D_n * 2
if __name__ == '__main__':
    gamma_list = [3.88295274,  3.58216713, 3.06842937, 2.22952243, 0.54787087]
    thickness_list = [105.1,	94.238,	82.856,	48.264,	0]

    break_point = [0, 1.85,	5.08,	8.34,	11.705,	14.5]
    airfoil_root = 'dae31'
    airfoil_tip = 'dae41'

    x = PSO('dae31_to_dae41', 9, 2.5, 0.00001604, gamma_list, thickness_list, airfoil_root, airfoil_tip, 39)
    result = x.execute()
    chord_list, mix_list, gamma_zure_list, drag_list, thickness_list_result = result
    chord_list.insert(1, chord_list[0])
    mix_list.insert(1, mix_list[0])
    calc_cd = Calcurate_Cd(
        break_point, chord_list, mix_list, 'dae31_to_dae41', 9, 2.5, 0.00001604,
        gamma_list, thickness_list, airfoil_root, airfoil_tip, 39
    )
    total_cd = calc_cd.calcularate_cd()
    print('翼弦長:', chord_list)
    print('翼型混合率:', mix_list)
    print('ガンマずれリスト:', gamma_zure_list)
    print('最小厚さリスト:', thickness_list_result)
    print('形状抗力:', total_cd)

    airfoil = Merge_Airfoil(airfoil_root, airfoil_tip, mix_list)
    airfoil_df = airfoil.merge()

    span = break_point[-1] * 2
    # breakpoint_number = 7
    # wing_length = span / breakpoint_number
    # station0 = 0
    # station1 = wing_length / 2
    # station2 = station1 + wing_length
    # station3 = station2 + wing_length
    # station4 = station3 + wing_length - 1
    # station5 = span / 2
    # print(station0, station1, station2, station3, station4, station5)
    # break_point = [station0, station1, station2, station3, station4, station5]

    area = np.zeros(100)
    delta_y = break_point[-1] / 100
    y = np.linspace(0, break_point[-1], 101)
    chord_fx = interpolate.interp1d(break_point, chord_list)
    for i2 in range(100):
        area[i2] = (chord_fx(y[i2]) + chord_fx(y[i2 + 1])) * delta_y / 2
    wing_area = np.sum(area) * 2
    aspect_ratio = (span ** 2) / wing_area
    print('翼面積 :' + str(wing_area))
    print('アスペクト比 :' + str(aspect_ratio))




    b = chord_fx(y)
    front = b * 0.31
    rear = b * (1 - 0.31) * -1
    fig, ax = plt.subplots()
    ax.plot(y, front, color='r')
    ax.plot(y, rear, color='r')
    ax.set_aspect('equal')
    ax.set_title('wing')
    plt.show()

    # データの定義
    data = {
        'y': break_point,
        'chord': chord_list,
        'offset': [0, 0, 0, 0, 0, 0],
        'dihedral': [0, 1, 2, 5, 6, 0],
        'twist': [0, 0, 0, 0, 0, 0],
        'xpanel': [13, 13, 13, 13, 13, 13],
        'ypanel': [19, 19, 19, 19, 19, 2],
        'Column8': [1, 1, 1, 1, 1, 1],
        'Column9': [-2, 0, 0, 0, 0, 0],
        'airfoil1': ['airfoil', 'airfoil', 'airfoil', 'airfoil', 'airfoil', 'airfoil'],
        'airfoil2': ['airfoil', 'airfoil', 'airfoil', 'airfoil', 'airfoil', 'airfoil']
    }
    print(break_point, chord_list)
    # DataFrameの作成
    df_xwimp = pd.DataFrame(data)
    output_file_name = 'pso_9_29.' + '.xwimp'
    df_xwimp.to_csv(output_file_name, index=False, sep=' ')
