import os
import json_tricks as jsont
from gyro_analysis import local_path as lp
from gyro_analysis.shotinfo import ShotInfo
import pandas as pd
import numpy as np


class ShotFrame:
    def __init__(self, run_number, shot_number, amp_lb=0, amp_ub=5):
        self.run_number = run_number
        self.shot_number = shot_number
        self.amp_lb = amp_lb
        self.amp_ub = amp_ub
        self.shd = self.load_shd()
        labels, det_labels, dark_labels = self.extract_labels()
        self.labels = labels
        self.det_labels = det_labels
        self.dark_labels = dark_labels
        self.shot_data_frame = self.build_data_frame()
        self.shot_info_frame = self.build_info_frame()
        self.shot_frame = self.build_shot_frame()
        self.unpack_fkeys = ['H', 'N', 'X', 'CP']
        self.unpack_dicts(['freqs', 'freq_errs'], self.unpack_fkeys)
        self.unpack_amps(amp_lb, amp_ub, self.unpack_fkeys)

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

    def extract_labels(self):
        labels = list(self.shd.keys())
        labels.remove('shotinfo')
        det_labels = [l for l in labels if len(l) == 1]
        dark_labels = [l for l in labels if len(l) == 2]
        return labels, det_labels, dark_labels

    def build_shot_frame(self):
        shot_data_frame = self.build_data_frame()
        shot_info_frame = self.build_info_frame()
        shot_frame = pd.concat([shot_data_frame, shot_info_frame], axis=1)
        shot_frame = shot_frame.set_index('label')
        return shot_frame

    def build_data_frame(self):
        shot_data_list = []
        for l in self.labels:
            data_dict = dict(self.shd[l])
            data_dict['label'] = l
            if 'freq' in data_dict:
                data_dict['freqs'] = data_dict.pop('freq')
            if 'freq_err' in data_dict:
                data_dict['freq_errs'] = data_dict.pop('freq_err')
            if 'amp' in data_dict:
                data_dict['amps'] = data_dict.pop('amp')
            if 'run_number' in data_dict:
                data_dict.pop('run_number')
            if 'shot_number' in data_dict:
                data_dict.pop('shot_number')
            shot_data_list.append(pd.Series(data_dict))
        shot_data_frame = pd.concat(shot_data_list, axis=1, sort=True).transpose()
        return shot_data_frame

    def build_info_frame(self):
        shot_info = pd.DataFrame(pd.Series(ShotInfo(self.run_number, self.shot_number).__dict__)).transpose()
        shot_info_frame = pd.DataFrame(np.repeat(shot_info.values, len(self.labels), axis=0))
        shot_info_frame.columns = shot_info.columns
        return shot_info_frame

    def unpack_dicts(self, dict_names, keys):
        dict_names = list(dict_names)
        keys = list(keys)
        for dict_name in dict_names:
            data_dict = self.shot_frame[dict_name]
            for key in keys:
                if dict_name[-1] == 's':
                    col_name = dict_name[0:-1] + '_' + key
                else:
                    col_name = dict_name + '_' + key
                self.shot_frame[col_name] = pd.Series([data_dict[j][key] for j in range(len(data_dict))], self.labels)

    def unpack_amps(self, lb, ub, keys):
        if 'CP' in keys:
            keys.remove('CP')
        data = self.shot_frame['amps']
        for key in keys:
            col_name = 'amp' + '_' + key
            det_series = pd.Series([np.mean(data[l][key][lb:ub]) for l in self.det_labels], self.det_labels)
            dark_series = pd.Series([np.nan for l in self.dark_labels], self.dark_labels)
            self.shot_frame[col_name] = pd.concat([det_series, dark_series])
