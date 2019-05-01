import numpy as np
from matplotlib import pyplot as plt
import os
import json_tricks as jsont
from gyro_analysis.local_path import shotdir
from string import ascii_lowercase
import pandas as pd


class PdRunAnalyzer:
    def __init__(self, run_number, sequence_name):
        self.run_number = str(int(run_number)).zfill(4)
        self.sequence_name = str(sequence_name)
        self.shotdir_run = os.path.join(shotdir, self.run_number)
        self.file_list = sorted(os.listdir(self.shotdir_run))
        self.shot_data = self.load_shot_data()
        self.n_det = int(len(self.shot_data[0].keys()) / 2)
        self.det_labels = list(ascii_lowercase[:self.n_det])
        self.dark_labels = [self.det_labels[i] + self.det_labels[i+1] for i in np.arange(self.n_det - 1)]
        self.flag = (self.shot_data[0]['shotinfo']['exists'] == True)
        self.fkeys = self.shot_data[0][self.det_labels[0]]['fkeys']
        self.bl = self.shot_data[0][self.det_labels[0]]['bl']
        [freqs, freq_err] = self.collect_freqs_freqerr()
        self.freqs = freqs
        self.freq_errs = freq_err
        self.chi_square = self.calc_chi_sqaure()
        self.T2 = self.collect_t2()
        self.amps = self.collect_amps()
        self.residuals = self.collect_residuals()
        if self.flag:
            self.shot_number = [i['shotinfo']['shot_number'] for i in self.shot_data]
            self.shot_number_int = list(map(int, self.shot_number))
            self.waveform_config = [i['shotinfo']['waveform_config'] for i in self.shot_data]
            self.cycle_number = [i['shotinfo']['cycle_number'] for i in self.shot_data]
            self.cycle_total = len(set(self.cycle_number))
            self.sequence = [i['shotinfo']['sequence_var'] for i in self.shot_data]
            self.sequence_per_cycle = self.get_sequence_per_cycle()
            [fpc, fepc] = self.get_freqs_per_cycle()
            self.freqs_per_cycle = fpc
            self.freq_errs_per_cycle = fepc
            self.amps_per_cycle = self.get_amps_per_cycle()
        else:
            self.shot_number = [i.split('.')[0].split('_')[1] for i in self.file_list]
            self.shot_number_int = list(map(int, self.shot_number))
            self.he_angle = None
            self.ne_angle = None
            self.xe_angle = None

    def load_shot_data(self):
        """Load all shot data files in the same run specified by run_number and return a shot_data list"""
        file_list = self.file_list
        shot_data = []
        for i in range(len(file_list)):
            file_path = os.path.join(self.shotdir_run, file_list[i])
            with open(file_path, 'r') as read_data:
                shot_data.append(jsont.load(read_data))
        return shot_data

    def build_df(self):
        df = pd.DataFrame(self.shot_data)

        df['shot_number'] = pd.Series([df['shotinfo'][i]['shot_number'] for i in range(len(df))])
        df['cycle_number'] = pd.Series([df['shotinfo'][i]['cycle_number'] for i in range(len(df))])
        df['he_angle'] = pd.Series([df['shotinfo'][i]['he_angle'] for i in range(len(df))])
        df['ne_angle'] = pd.Series([df['shotinfo'][i]['ne_angle'] for i in range(len(df))])
        df['xe_angle'] = pd.Series([df['shotinfo'][i]['xe_angle'] for i in range(len(df))])
        df['timestamp'] = pd.Series([df['shotinfo'][i]['timestamp'] for i in range(len(df))])
