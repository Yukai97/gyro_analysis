import math
import os
from matplotlib import pyplot as plt
import pandas as pd
from gyro_analysis.local_path import paths as pt
from gyro_analysis.shot_frame import ShotFrame
shotdir = pt.shotdir


class RunDataFrame:

    def __init__(self, run_number=None, shot_frame_list=None, amp_lb=0, amp_ub=5):
        self.amp_lb = amp_lb
        self.amp_ub = amp_ub
        self.fkeys = ['CP', 'H', 'N', 'X']
        self.markers = {'H': '^', 'N': 'v', 'X': 'x', 'CP': 'o'}
        if run_number:
            self.shot_frame_list = self.build_shot_frame_list(run_number)
        else:
            self.shot_frame_list = shot_frame_list
        run_number, labels, run_frame = self.build_run_frame()
        self.run_number = run_number
        self.labels = labels
        self.det_labels = [l for l in self.labels if len(l) == 1]
        self.det_labels.sort()
        self.dark_labels = [l for l in self.labels if len(l) == 2]
        self.dark_labels.sort()
        self.run_frame = run_frame

    def build_shot_frame_list(self, run_number):
        if type(run_number) == str or type(run_number) == int:
            run_number = str(int(run_number)).zfill(4)
            run_number = [run_number]
        shot_frame_list = []
        for i in run_number:
            shotdir_run = os.path.join(shotdir, i)
            file_list = sorted(os.listdir(shotdir_run))
            for file in file_list:
                shot_number = file.split('.')[0].split('_')[1]
                shot_frame_list.append(ShotFrame(i, shot_number, self.amp_lb, self.amp_ub))
        return shot_frame_list

    def build_run_frame(self):
        shot_frames = [self.shot_frame_list[i].shot_frame for i in range(len(self.shot_frame_list))]
        run_frame = pd.concat(shot_frames)
        run_frame = run_frame.reset_index()
        run_number = list(set(run_frame['run_number']))
        labels = list(set(run_frame['label']))
        run_frame = run_frame.set_index(['run_number', 'shot_number', 'label'])
        return run_number, labels, run_frame

    def plot_freqs_shot(self):
        run_frame = self.run_frame.reset_index()
        run_frame['shot_number'] = pd.to_numeric(run_frame['shot_number'])
        run_frame = run_frame.set_index(['run_number', 'label', 'shot_number'])

        fkey_num = len(self.fkeys)
        col = 2
        row = math.ceil(fkey_num / col)
        figsize = (col * 6, row * 4)
        wspace = 0.25
        hspace = 0.4
        fontsize = 14

        for i in self.run_number:
            fig, axes = plt.subplots(row, col, figsize=figsize)
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
            fig.suptitle('Run ' + i + ' light')
            for j in range(len(self.fkeys)):
                key = self.fkeys[j]
                freq_name = 'freq_' + key
                freq_err_name = 'freq_err_' + key
                ax = axes[int(j / col)][j % col]
                for l in self.det_labels:
                    error_bar = run_frame[freq_err_name][i][l]
                    run_frame[freq_name][i][l].plot(yerr=error_bar, ax=ax, fmt='-' + self.markers[key])
                ax.set_xlabel('shot number')
                ax.set_ylabel('$\omega_{' + key + '}$')
                ax.legend(self.det_labels, fontsize=fontsize)
            if self.dark_labels:
                fig, axes = plt.subplots(row, col, figsize=figsize)
                plt.subplots_adjust(wspace=wspace, hspace=hspace)
                fig.suptitle('Run ' + i + ' dark')
                for j in range(len(self.fkeys)):
                    key = self.fkeys[j]
                    freq_name = 'freq_' + key
                    freq_err_name = 'freq_err_' + key
                    ax = axes[int(j / col)][j % col]
                    for l in self.dark_labels:
                        error_bar = run_frame[freq_err_name][i][l]
                        run_frame[freq_name][i][l].plot(yerr=error_bar, ax=ax, fmt='-' + self.markers[key])
                    ax.set_xlabel('shot number')
                    ax.set_ylabel('$\omega_{' + key + '}$')
                    ax.legend(self.dark_labels, fontsize=fontsize)

    def plot_amps_shot(self):
        run_frame = self.run_frame.reset_index()
        run_frame['shot_number'] = pd.to_numeric(run_frame['shot_number'])
        run_frame = run_frame.set_index(['run_number', 'label', 'shot_number'])
        fkeys = self.fkeys
        if 'CP' in fkeys:
            fkeys.remove('CP')
        fontsize = 14
        for i in self.run_number:
            plt.figure()
            legend = []
            plt.title('Run ' + i)
            for key in fkeys:
                amp_name = 'amp_' + key
                for l in self.det_labels:
                    legend.append(key + ' ' + l)
                    run_frame[amp_name][i][l].plot(style='-' + self.markers[key])
            plt.xlabel('Shot number')
            plt.ylabel('Amplitude [V]')
            plt.legend(legend, fontsize=fontsize)

    def plot_abs_res_max(self):
        run_frame = self.run_frame.reset_index()
        run_frame['shot_number'] = pd.to_numeric(run_frame['shot_number'])
        run_frame = run_frame.set_index(['run_number', 'label', 'shot_number'])
        for i in self.run_number:
            plt.figure()
            plt.title('Run ' + i)
            for l in self.det_labels:
                run_frame['abs_res_max'][i][l].plot(style='-o')
            plt.xlabel('Shot number')
            plt.ylabel('Abs res max')
            plt.legend(self.run_number)