import os
import json_tricks as jsont
from gyro_analysis import local_path as lp
from gyro_analysis.shotinfo import ShotInfo
import pandas as pd


class ShotSeries:
    def __init__(self, run_number, shot_number):
        self.run_number = run_number
        self.shot_number = shot_number
        self.shd = self.load_shd()
        labels = list(self.shd.keys())
        labels.remove('shotinfo')
        self.labels = labels
        self.shot_data_series = self.build_data_series()
        self.shot_info_series = pd.Series(ShotInfo(run_number, shot_number).__dict__)
        self.shot_series = pd.concat([self.shot_data_series, self.shot_info_series])

    def load_shd(self):
        run_number = str(self.run_number).zfill(4)
        shot_number = str(self.shot_number).zfill(3)
        shotdir_run = os.path.join(lp.shotdir, run_number)
        file_path = os.path.join(shotdir_run, run_number + '_' + shot_number + lp.shot_ex_data)
        try:
            with open(file_path, 'r') as read_data:
                shot_data = jsont.load(read_data)
        except FileNotFoundError:
            print('Do not find ' + file_path)
            shot_data = {}
        return shot_data

    def build_data_series(self):
        shot_data_series = {}
        for l in self.labels:
            shot_data_series[l] = pd.Series(self.shd[l])
        return pd.Series(shot_data_series)
