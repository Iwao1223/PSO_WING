import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from airfoil_interpolate import AirfoilInterpolate

class WingPlotter:
    def __init__(self, airfoil_root, break_point, chord_list, mix_list):
        self.airfoil_root = airfoil_root
        self.break_point = break_point
        self.chord_list = chord_list
        self.mix_list = mix_list

    def airfoil_shape_func(self, mix):
        dir_name = 'airfoil_data/FOILS'
        airfoil = self.airfoil_root + '.dat'
        path = Path.cwd()
        airfoil_file = path / dir_name / airfoil
        airfoil_df = pd.read_csv(
            airfoil_file, header=None, delim_whitespace=True,
            skipinitialspace=True, skiprows=1, dtype='float16'
        )
        airfoil_fx = AirfoilInterpolate(airfoil_df, 1, 0)
        airfoil_top_fx, airfoil_bottom_fx = airfoil_fx.airfoil_interpolate()
        x = np.linspace(0.01, 0.98, 100)
        z_top = airfoil_top_fx(x)
        z_bottom = airfoil_bottom_fx(x)
        z = np.concatenate([z_top, z_bottom[::-1]])
        x_full = np.concatenate([x, x[::-1]])
        return x_full, z

    def plot_surface(self, filename='wing_surface.png'):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        x_sections, y_sections, z_sections = [], [], []
        max_mix = max(self.mix_list)
        min_mix = min(self.mix_list)
        norm = plt.Normalize(min_mix, max_mix)
        cmap = plt.cm.rainbow

        for i, y in enumerate(self.break_point):
            x_af, z_af = self.airfoil_shape_func(self.mix_list[i])
            x_af = x_af * self.chord_list[i] * 1000
            z_af = z_af * self.chord_list[i] * 1000
            x_af = -x_af + self.chord_list[i] * 1000 * 0.39
            y_af = np.full_like(x_af, y * 1000)
            color = cmap(norm(self.mix_list[i]))
            x_sections.append(x_af)
            y_sections.append(y_af)
            z_sections.append(z_af)
            ax.plot(x_af, y_af, z_af, color=color, linewidth=0.8, alpha=0.7)
            ax.plot(x_af, -y_af, z_af, color=color, linewidth=0.8, alpha=0.7)
        X = np.array(x_sections)
        Y = np.array(y_sections)
        Z = np.array(z_sections)
        mix_array = np.array(self.mix_list)
        mix_colors = cmap(norm(mix_array))
        facecolors = np.tile(mix_colors[:, np.newaxis, :], (1, X.shape[1], 1))
        ax.plot_surface(X, Y, Z, facecolors=facecolors, alpha=0.85, rstride=1, cstride=1, linewidth=0, antialiased=True)
        ax.plot_surface(X, -Y, Z, facecolors=facecolors, alpha=0.85, rstride=1, cstride=1, linewidth=0, antialiased=True)
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor((0., 0., 0., 0.))
        ax.yaxis.pane.set_edgecolor((0., 0., 0., 0.))
        ax.zaxis.pane.set_edgecolor((0., 0., 0., 0.))
        ax.xaxis.pane.set_facecolor((1., 1., 1., 0.))
        ax.yaxis.pane.set_facecolor((1., 1., 1., 0.))
        ax.zaxis.pane.set_facecolor((1., 1., 1., 0.))
        x_range = X.max() - X.min()
        y_range = Y.max() - Y.min()
        z_range = Z.max() - Z.min()
        ax.set_box_aspect([x_range/1000, y_range*2/1000, z_range/1000])
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95)
        ax.view_init(elev=18, azim=-70)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

# --- 利用例 ---
if __name__ == '__main__':
    airfoil_root = 'dae31'
    break_point = [0, 1.85, 5.08, 8.34, 11.705, 14.5]
    chord_list = [1.0, 1.0, 0.8, 0.6, 0.4, 0.2]
    mix_list = [0, 20, 40, 60, 80, 100]
    plotter = WingPlotter(airfoil_root, break_point, chord_list, mix_list)
    plotter.plot_surface()