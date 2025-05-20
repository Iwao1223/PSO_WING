from airfoil_interpolate import AirfoilInterpolate
import pandas as pd
from pathlib import Path

class AirfoilThickness:
    """
    Class to calculate the thickness of an airfoil.
    """

    def __init__(self, airfoil_name, loc):
        self.airfoil_name = airfoil_name
        self.dir_name = 'airfoil_data/FOILS'
        self.airfoil = self.airfoil_name + '.dat'
        self.partition = 1000
        self.loc = int(loc / 100 * self.partition)
    def airfoil_thickness(self):

        """
        Read the airfoil data file.
        """

        path = Path.cwd()
        airfoil_file = path / self.dir_name / self.airfoil
        airfoil_df = pd.read_csv(airfoil_file, header=None, delim_whitespace=True, skipinitialspace=True, skiprows=1,
                          dtype='float16')
        airfoil_fx = AirfoilInterpolate(airfoil_df, self.partition, 0)
        airfoil_top_fx, airfoil_bottom_fx = airfoil_fx.airfoil_interpolate()
        """
        Calculate the max thickness of the airfoil.
        """
        y_diff = []
        range_x = int(self.partition * 0.9)
        for i in range(5, range_x):
            y_top = airfoil_top_fx(i)
            y_bottom = airfoil_bottom_fx(i)
            y_diff.append(y_top - y_bottom)
        max_thickness = max(y_diff)
        max_thickness_par = max(y_diff) / self.partition
        max_thickness_loc_par = y_diff.index(max_thickness) / self.partition
        thickness_loc_par = y_diff[self.loc] / self.partition

        return max_thickness_par, max_thickness_loc_par, thickness_loc_par

if __name__ == '__main__':
    x = AirfoilThickness('dae41', 50)
    max_thickness, max_thickness_loc, thickness_loc = x.airfoil_thickness()
    print(max_thickness, max_thickness_loc, thickness_loc)



