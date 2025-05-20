import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd


class AirfoilInterpolate:
    def __init__(self,airfoil_df,chord,offset):
        self.airfoil_df = airfoil_df
        self.chord = chord
        self.offset = offset

    def remove_duplicates(self, x, y):
        x, idx = np.unique(x, return_index=True)
        y = y[idx]
        return x, y

    def airfoil_interpolate(self):
        #airfoil = 'DAE31.dat'
        #airfoil_df = pd.read_table(airfoil, sep='\t', skiprows=1, header=None, engine='python')
        #print(airfoil_df)
        if type(self.airfoil_df.iloc[0, 0]) == str:
            #print('drop')
            airfoil_df_drop = self.airfoil_df.drop(self.airfoil_df.index[0])
        else:
            airfoil_df_drop = self.airfoil_df
        airfoil_df_float = airfoil_df_drop.astype('float64')
        airfoil_df_float_reset = airfoil_df_float.reset_index(drop=True)
        # airfoil_d_df = airfoil_df[airfoil_df[0] < self.max_airfoil_thickness_at[x] + 0.1].reset_index(drop=True)
        # border = airfoil_df_float[airfoil_df_float[1]>=0].min()
        border_idx = airfoil_df_float_reset[0].idxmin()
        border = airfoil_df_float_reset[0].min()
        if airfoil_df_float_reset[airfoil_df_float_reset[0] == border].shape[0] != 1:
            airfoil_df_float_min = airfoil_df_float_reset[airfoil_df_float_reset[0] == border]
            border_idx = airfoil_df_float_min[1].idxmin()
            #print('fix')
        #print(border_idx)
        airfoil_top_df, airfoil_bottom_df = airfoil_df_float_reset[:border_idx], airfoil_df_float_reset[border_idx:]
        # print(airfoil_top_df)
        airfoil_x_top_df, airfoil_y_top_df, airfoil_x_bottom_df, airfoil_y_bottom_df = airfoil_top_df[0], \
            airfoil_top_df[1], airfoil_bottom_df[0], airfoil_bottom_df[1]
        airfoil_x_top, airfoil_y_top, airfoil_x_bottom, airfoil_y_bottom = airfoil_x_top_df.to_numpy(), airfoil_y_top_df.to_numpy(), airfoil_x_bottom_df.to_numpy(), airfoil_y_bottom_df.to_numpy()
        airfoil_x_top_mm, airfoil_y_top_mm, airfoil_x_bottom_mm, airfoil_y_bottom_mm = airfoil_x_top * self.chord + self.offset, airfoil_y_top * self.chord, airfoil_x_bottom * self.chord + self.offset, airfoil_y_bottom * self.chord
        #print(airfoil_x_top_mm)
        airfoil_x_bottom_mm, airfoil_y_bottom_mm = self.remove_duplicates(airfoil_x_bottom_mm, airfoil_y_bottom_mm)
        airfoil_up_fx, airfoil_down_fx = interpolate.interp1d(airfoil_x_top_mm, airfoil_y_top_mm,
                                                              kind='cubic'), interpolate.interp1d(airfoil_x_bottom_mm,
                                                                                                   airfoil_y_bottom_mm,
                                                                                                   kind='cubic')
        # x_up, x_down = np.linspace(0, self.chord, self.airfoil_number_partition), np.linspace(0, self.chord,self.airfoil_number_partition)
        # y_up = airfoil_up_fx(x_up)
        # y_down = airfoil_down_fx(x_down)
        # fig, ax = plt.subplots(figsize=(40, 5))
        # plt.plot(x_up, airfoil_up_fx(x_up))
        return airfoil_up_fx, airfoil_down_fx


if __name__ == '__main__':
    x = AirfoilInterpolate(800,100)
    x.airfoil_interpolate()