# -*- coding: utf-8 -*-
# Cording by Sogo Iwao from nasg25(m42) on 2024-5-22.

from merge_airfoil import Merge_Airfoil

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

    def read_data(self):
        print('---Read_start---')
        start = time.time()
        path = Path.cwd()
        xflr = path / self.dir
        xflr_g = xflr.iterdir()
        paths = [str(z) for z in xflr_g]

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
        print(self.df)
        end = time.time()
        print('処理時間:', end - start)
        print('---Read_end---')
        return self.df, self.Re_max

    def interpolate_cd(self):
        df_alpha = self.df[self.df['alpha'] == self.alpha]
        df_alpha_sort = df_alpha.sort_values(['Re', 'mix'])
        df_alpha_sort['Re*Cd'] = df_alpha_sort['Re'] * df_alpha_sort['Cd']
        # df_alpha_sort['Re*Cl'] = df_alpha_sort['Re'] * df_alpha_sort['Cl']
        df_alpha_sort_Re_Cd = df_alpha_sort[['Re', 'mix', 'Re*Cd']]
        df_alpha_sort_Re_Cd_drop_index = df_alpha_sort_Re_Cd.index[df_alpha_sort_Re_Cd['Re'] % 5000 != 0]
        df_alpha_sort_Re_Cd_bug_bug = df_alpha_sort_Re_Cd.drop(df_alpha_sort_Re_Cd_drop_index)
        df_alpha_sort_Re_Cd_re_bug = df_alpha_sort_Re_Cd_bug_bug.set_index(['Re', 'mix'])
        df_alpha_sort_Re_Cd_re = df_alpha_sort_Re_Cd_re_bug.drop_duplicates()
        df_alpha_sort_Re_Cd_unstack = df_alpha_sort_Re_Cd_re.unstack()
        # print(df_alpha_sort_Re_Cd_unstack )
        df_alpha_sort_Re_Cd_unstack_interpolate = df_alpha_sort_Re_Cd_unstack.interpolate()

        foil_alpha_Re_Cd = df_alpha_sort_Re_Cd_unstack_interpolate.to_numpy()
        np.set_printoptions(threshold=1000)
        # print(np.get_printoptions())
        print(foil_alpha_Re_Cd)
        # 行の表示を省略しない
        # pd.set_option('display.max_rows', 100)

        # 列の表示を省略しない
        # pd.set_option('display.max_columns', none)

        # print(df_alpha_sort_Re_Cd_unstack_interpolate )
        # print(foil_alpha_Re_Cd)
        # 解析データのRe数の幅と間隔
        x = range(self.Re_min, self.Re_max + 1, 5000)
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
        df_alpha_sort_Re_Cl_drop_index = df_alpha_sort_Re_Cl.index[df_alpha_sort_Re_Cl['Re'] % 5000 != 0]
        df_alpha_sort_Re_Cl_bug_bug = df_alpha_sort_Re_Cl.drop(df_alpha_sort_Re_Cl_drop_index)
        df_alpha_sort_Re_Cl_re_bug = df_alpha_sort_Re_Cl_bug_bug.set_index(['Re', 'mix'])
        df_alpha_sort_Re_Cl_re = df_alpha_sort_Re_Cl_re_bug.drop_duplicates()
        df_alpha_sort_Re_Cl_unstack = df_alpha_sort_Re_Cl_re.unstack()
        # print(df_alpha_sort_Re_Cl_unstack )
        df_alpha_sort_Re_Cl_unstack_interpolate = df_alpha_sort_Re_Cl_unstack.interpolate()
        foil_alpha_Re_Cl = df_alpha_sort_Re_Cl_unstack_interpolate.to_numpy()

        # 解析データのRe数の幅と間隔
        x = range(self.Re_min, self.Re_max + 1, 5000)
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
    def __init__(self, dir2, U, alpha, nu, gamma_object):
        self.dir = dir2
        self.alpha = alpha
        self.U = U
        self.nu = nu
        self.gamma_list = gamma_object
        read = Read(self.dir, self.alpha)
        self.df, self.Re_max = read.read_data()
        self.fx_cd = read.interpolate_cd()
        self.fx_cl = read.interpolate_gamma()
        self.chord_list = []
        self.mix_list = []
        self.drag_list = []
        self.gamma_zure_list = []

    # def read_fx(self):
    # read = Read(self.dir,self.alpha)
    # read.read_data()
    # self.fx_cd = read.interpolate_cd()
    # self.fx_cl = read.interpolate_gamma()

    def fitness(self, Re, m):
        # print(Re)
        z = self.fx_cd((Re, m))
        return z

    def constraints(self, Re, m):
        gamma = self.fx_cl((Re, m)) * self.nu / 2
        mu = abs(1 - gamma / self.gamma_object)
        return mu

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
        print('---PSO_start---')
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
        print(global_best_position)
        print(np.min(personal_best_scores))
        print(x.constraints(global_best_position["x1"], global_best_position["x2"]))
        print(global_best_position["x1"] * self.nu / self.U)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection='3d')
        x1_coord = np.linspace(x1_min, x1_max, 100)
        x2_coord = np.linspace(x2_min, x2_max, 100)
        X, Y = np.meshgrid(x1_coord, x2_coord)
        ax1.plot_wireframe(X, Y, x.fitness(X, Y), color='b', rstride=2, cstride=2, linewidth=0.3)
        ax1.set_xlabel('$Re$')
        ax1.set_ylabel('$m$')
        ax1.set_zlabel('$Re*Cd$')
        ax1.scatter3D(global_best_position['x1'], global_best_position['x2'], np.min(personal_best_scores), color='red')

        return (global_best_position['x1'] * self.nu / self.U, global_best_position['x2'],
                x.constraints(global_best_position["x1"], global_best_position["x2"]), np.min(personal_best_scores))

    def execute(self):
        for i1 in self.gamma_list:
            zure = 1
            while zure > 0.001:
                self.gamma_object = i1
                output = self.pso()
                zure = output[2]
            self.chord_list.append(output[0])
            self.mix_list.append(output[1])
            self.gamma_zure_list.append(output[2])
            self.drag_list.append(output[3])
        return self.chord_list,self.mix_list,self.gamma_zure_list,self.drag_list

if __name__ == '__main__':
    gamma_list = [3.688137646092975, 3.357820667712207, 2.6395343114078194, 1.4533089846146512, 0.44782811326941674]
    x = PSO('peg32_to_revT', 9, 3, 0.00001604, gamma_list)
    result = x.execute()
    chord_list, mix_list, gamma_zure_list, drag_list = result
    chord_list.insert(1, chord_list[0])
    print(chord_list)
    print(mix_list)
    print(gamma_zure_list)
    print(drag_list)
    print('total_drag:{}'.format(sum(drag_list)))

    airfoil_root = 'airfoil_date/FOILS/pegasus32'
    airfoil_tip = 'airfoil_date/FOILS/rev_tip_115_mod'

    airfoil = Merge_Airfoil(airfoil_root, airfoil_tip,mix_list)
    airfoil_df = airfoil.merge()


    span = 27.5
    breakpoint_number = 7
    wing_length = span / breakpoint_number
    station0 = 0
    station1 = wing_length / 2
    station2 = station1 + wing_length
    station3 = station2 + wing_length
    station4 = station3 + wing_length - 1
    station5 = span / 2
    area = np.zeros(100)
    print(station0, station1, station2, station3, station4, station5)
    break_point = [station0, station1, station2, station3, station4, station5]
    delta_y = station5 / 100
    y = np.linspace(0, station5, 101)
    chord_fx = interpolate.interp1d(break_point, chord_list)
    # thickness_mm = self.chord * self.airfoil_thickness
    # thickness_fx = interpolate.interp1d(self.breakpoint,self.thickness_mm)
    # print(self.chord_fx(0))
    # print(self.delta_y)
    # print(self.y)
    for i2 in range(100):
        area[i2] = (chord_fx(y[i2]) + chord_fx(y[i2 + 1])) * delta_y / 2
        # print(self.area)
    print('wing area :' + str(np.sum(area) * 2))

    b = chord_fx(y)
    front = b * 0.31
    rear = b * (1 - 0.31) * -1
    fig, ax = plt.subplots()
    ax.plot(y, front, color='r')
    ax.plot(y, rear, color='r')
    ax.set_aspect('equal')
    ax.set_title('翼平面形')

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
    output_file_name = 'pso_9_25' + '.xwimp'
    df_xwimp.to_csv(output_file_name, index=False, sep=' ')
