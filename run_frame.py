import math
import numpy as np
import statsmodels.api as sm
import os
import sys
from matplotlib import pyplot as plt
import pandas as pd
from gyro_analysis.local_path import paths as pt
from gyro_analysis.shot_frame import ShotFrame
from copy import deepcopy
shotdir = pt.shotdir
from gyro_analysis import mu_0, gamma_ne
import statsmodels.formula.api as smf


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
                    x_axis = list(run_frame[freq_name][i][l].index)
                    ax.set_xlabel('shot number')
                    ax.set_ylabel('$\omega_{' + key + '}$')
                    ax.set_xlim(min(x_axis)-2, max(x_axis)+2)
                    ax.legend(self.dark_labels, fontsize=fontsize)

    def plot_amps_shot(self):
        run_frame = self.run_frame.reset_index()
        run_frame = run_frame.set_index(['run_number', 'label', 'shot_number'])
        fkeys = deepcopy(self.fkeys)
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
            x_axis = list(run_frame[amp_name][i][l].index)
            plt.xlabel('Shot number')
            plt.ylabel('Amplitude [V]')
            plt.xlim(min(x_axis)-2, max(x_axis)+2)
            plt.legend(legend, fontsize=fontsize)

    def plot_abs_res_max(self):
        run_frame = self.run_frame.reset_index()
        run_frame = run_frame.set_index(['run_number', 'label', 'shot_number'])
        for i in self.run_number:
            plt.figure()
            plt.title('Run ' + i)
            for l in self.det_labels:
                run_frame['abs_res_max'][i][l].plot(style='-o')
            x_axis = list(run_frame['abs_res_max'][i][l].index)
            plt.xlabel('Shot number')
            plt.ylabel('Abs res max')
            plt.xlim(min(x_axis)-2, max(x_axis)+2)
            plt.legend(self.det_labels)

    def plot_freqs_sequence(self, legend_switch=True):
        run_frame = self.run_frame.reset_index()
        fkey_num = len(self.fkeys)
        col = 2
        row = math.ceil(fkey_num / col)
        figsize = (col * 6, row * 4)
        wspace = 0.25
        hspace = 0.4
        fontsize = 14
        for i in self.run_number:
            cycle_number = list(set(run_frame['cycle_number']))
            run_frame = run_frame.set_index(['run_number', 'label', 'cycle_number', 'sequence_var'])
            for l in self.labels:
                fig, axes = plt.subplots(row, col, figsize=figsize)
                plt.subplots_adjust(wspace=wspace, hspace=hspace)
                fig.suptitle('Run ' + i + ' ' + l)
                for j in range(len(self.fkeys)):
                    key = self.fkeys[j]
                    freq_name = 'freq_' + key
                    freq_err_name = 'freq_err_' + key
                    ax = axes[int(j / col)][j % col]
                    legend = []
                    for m in cycle_number:
                        cycle_index = str(m)
                        legend.append(cycle_index)
                        error_bar = run_frame[freq_err_name][i][l][cycle_index]
                        run_frame[freq_name][i][l][cycle_index].plot(yerr=error_bar, ax=ax, fmt='-' + self.markers[key])
                    x_axis = list(run_frame[freq_name][i][l][cycle_index].index)
                    ax.set_xlabel('Sequence var')
                    ax.set_ylabel('$\omega_{' + key + '}$')
                    ax.set_xlim(min(x_axis)-2, max(x_axis)+2)
                    if legend_switch:
                        ax.legend(legend, fontsize=fontsize)

    def calc_kappa(self, formula='freq_CP ~ M0 : np.cos(angle) + np.cos(angle)'):
        run_frame = self.run_frame.reset_index()
        run_frame = run_frame.set_index(['run_number', 'label'])
        kappa_res_list = []
        if 'M0' not in formula:
            print('The formula is incorrect!')
            sys.exit()
        for i in self.run_number:
            if pd.Series.equals(run_frame['ne_angle'][i], run_frame['sequence_var'][i]):
                seq_name = 'N'
            elif pd.Series.equals(run_frame['he_angle'][i], run_frame['sequence_var'][i]):
                seq_name = 'H'
            else:
                print('The sequence is incorrect!')
                sys.exit()
            if 'M0_H' in formula or 'M0_N' in formula:
                fit_formula = formula
            else:
                index = formula.find('M0') + 2
                fit_formula = formula[:index] + '_N' + formula[index:]
            for l in self.dark_labels:
                fit_frame = run_frame.loc[(i, l), ['freq_CP', 'freq_err_CP', 'M0_H', 'M0_N']]
                fit_frame['angle'] = run_frame.loc[(i, l), 'sequence_var'].apply(np.radians)
                model = smf.ols(formula=fit_formula, data=fit_frame)
                results = model.fit()
                params = results.params
                error_bars = results.bse
                if seq_name == 'N':
                    kappa_he_ne = -params[1]/gamma_ne/mu_0
                    kappa_he_ne_err = error_bars[1]/gamma_ne/mu_0
                else:
                    kappa_he_ne = params[1]/gamma_ne/mu_0
                    kappa_he_ne_err = error_bars[1]/gamma_ne/mu_0
                kappa_series = pd.Series({'run_number': i, 'label': l, 'seq_name': seq_name, 'kappa_he_ne': kappa_he_ne,
                                          'kappa_he_err': kappa_he_ne_err, 'fit_results': results,
                                          'fit_formula': fit_formula})
                kappa_res_list.append(kappa_series)
                residuals = results.resid
        kappa_data_frame = pd.DataFrame(kappa_res_list)
        return kappa_data_frame


