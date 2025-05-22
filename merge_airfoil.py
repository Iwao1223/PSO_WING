#-*- coding: utf-8 -*-
#Cording by Sogo Iwao from nasg25(m42) on 2024-6-11.


import numpy as np
import pandas as pd
from scipy import interpolate
from pathlib import Path

class Merge_Airfoil():
    def __init__(self,airfoil1,airfoil2,ratio_list):
        self.airfoil1_name = airfoil1
        self.airfoil2_name = airfoil2
        self.dir_name = 'airfoil_data/FOILS'
        self.airfoil1 = self.airfoil1_name + '.dat'
        self.airfoil2 = self.airfoil2_name + '.dat'
        self.ratio_list = ratio_list
        self.airfoil_name = self.airfoil1_name + '_' +'0' + '_' + self.airfoil2_name
        self.df_list = []
    def merge(self):
        
        path = Path.cwd()
        airfoil1_file = path / self.dir_name /self.airfoil1
        airfoil2_file = path / self.dir_name /self.airfoil2

        #print(airfoil1_file)
        df1 = pd.read_csv(airfoil1_file, header=None, delim_whitespace=True, skipinitialspace=True, skiprows=1, dtype='float16')
        df2 = pd.read_csv(airfoil2_file, header=None, delim_whitespace=True, skipinitialspace=True, skiprows=1, dtype='float16')
        
        x1_min = df1[0].idxmin()
        x2_min = df2[0].idxmin()
        
        x1_min_value = df1[0].min()
        x2_min_value = df2[0].min()
        if x1_min_value < x2_min_value:
            delta_x = x2_min_value - x1_min_value
            df1[0] = df1[0] + delta_x * 1.1
        
        self.x1 = df1[0].tolist()
        self.x1_top = df1[0][:x1_min].tolist()
        self.y1_top = df1[1][:x1_min].tolist()
        self.x1_bottom = df1[0][x1_min:].tolist()
        self.y1_bottom = df1[1][x1_min:].tolist()
    
        self.x2_top = df2[0][:x2_min+1].tolist()
        self.y2_top = df2[1][:x2_min+1].tolist()
        self.x2_bottom = df2[0][x2_min:].tolist()
        self.y2_bottom = df2[1][x2_min:].tolist()
        
    
        x2_top_re = list(reversed(self.x2_top))
        y2_top_re = list(reversed(self.y2_top))
        fx_bottom = interpolate.interp1d(self.x2_bottom,self.y2_bottom,kind='linear')
        fx_top = interpolate.interp1d(x2_top_re,y2_top_re,kind='linear')
        
        
        for i in self.ratio_list:
            self.airfoil_name = self.airfoil1_name + '_' +str(int(i)) + '_' + self.airfoil2_name
            muy_list = []
            for (ux, uy) in zip(self.x1_top,self.y1_top):
                tuy1 = fx_top(ux)
                muy1 = uy + ( (tuy1 - uy) * float(100-i) /100 )
                muy_list.append(muy1)
                
            for (lx, ly) in zip(self.x1_bottom,self.y1_bottom):
                tuy2 = fx_bottom(lx)
                muy2 = ly + ( (tuy2 - ly) * float(100-i) /100 )
                muy_list.append(muy2)
            list_x_y = [self.x1,muy_list]
            df_output1 = pd.DataFrame(list_x_y)
            df_output1.insert(0,'name',self.airfoil_name)
            df_output_T = df_output1.T
            df_output_bug = df_output_T.replace({1:{self.airfoil_name:''}})
            df_output = df_output_bug[ df_output_bug[0] != 0]
            self.df_list.append(df_output_bug)
            output_file_name = self.airfoil_name + '.dat'
            df_output.to_csv(output_file_name, header=False, index=False, sep=' ')
        return df_output_bug
            
if __name__ == '__main__':
    #a = range(0,101,5)
    a = [89.06303447460459, 87.77424021938518, 86.08447204631838, 63.329192108405906, 34.14530285418442, 4.159986467699512]
    airfoil_root = 'airfoil_data/FOILS/dae11'
    airfoil_tip = 'airfoil_data/FOILS/rev_tip_115_mod'

    airfoil = Merge_Airfoil('dae31', 'dae41', a)
    airfoil_df = airfoil.merge()

    #x = Merge_Airfoil('pegasus30','revT',a)
    #x.merge()





