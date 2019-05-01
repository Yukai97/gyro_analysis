import numpy as np
from matplotlib import pyplot as plt
import os
import json_tricks as jsont
from gyro_analysis.local_path import shotdir
from string import ascii_lowercase


class RunAnalyzer:
    """Analyze all shot data files in the same run specified by the run_number and plot related figures
    Args:
        run_number: specify the run we need to analyze
    """

    def __init__(self, run_number, sequence_name):
        self.run_number = str(int(run_number)).zfill(4)
        self.sequence_name = str(sequence_name)
        self.shotdir_run = os.path.join(shotdir, self.run_number)
        self.file_list = sorted(os.listdir(self.shotdir_run))
        self.shot_data = self.load_shot_data()
        self.n_det = int(len(self.shot_data[0].keys())/2)
        self.timestamp = [i['shotinfo']['timestamp'] for i in self.shot_data]
        self.flag = (self.shot_data[0]['shotinfo']['exists'] == True)
        self.det_labels = list(ascii_lowercase[:self.n_det])
        self.dark_labels = [self.det_labels[i] + self.det_labels[i+1] for i in np.arange(self.n_det - 1)]
        self.fkeys = self.shot_data[0][self.det_labels[0]]['fkeys']
        self.bl = self.shot_data[0][self.det_labels[0]]['bl']
        [freqs, freq_err] = self.collect_freqs_freqerr()
        self.freqs = freqs
        self.freq_errs = freq_err
        self.chi_square = self.calc_chi_sqaure()
        self.T2 = self.collect_t2()
        self.amps = self.collect_amps()
        self.residuals = self.collect_residuals()
        self.abs_res_max = self.collect_abs_res_max()
        if self.flag:
            self.shot_number = [i['shotinfo']['shot_number'] for i in self.shot_data]
            self.shot_number_int = list(map(int, self.shot_number))
            self.waveform_config = [i['shotinfo']['waveform_config'] for i in self.shot_data]
            self.he_angle = [i['shotinfo']['he_angle'] for i in self.shot_data]
            self.ne_angle = [i['shotinfo']['ne_angle'] for i in self.shot_data]
            self.xe_angle = [i['shotinfo']['xe_angle'] for i in self.shot_data]
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

    def collect_freqs_freqerr(self):
        """Collect all frequencies and corresponding errorbars and return freqs and freq_err dictionaries"""
        freqs = {}
        freq_errs = {}
        for label in self.det_labels:
            freqs[label] = {}
            freq_errs[label] = {}
            for f in self.fkeys:
                freqs[label][f] = np.array([i[label]['freq'][f] for i in self.shot_data])
                freq_errs[label][f] = np.array([i[label]['freq_err'][f] for i in self.shot_data])
        for label in self.dark_labels:
            freqs[label] = {}
            freq_errs[label] = {}
            for f in self.fkeys:
                freqs[label][f] = np.array([i[label]['freq'][f] for i in self.shot_data])
                freq_errs[label][f] = np.array([i[label]['freq_err'][f] for i in self.shot_data])
        return freqs, freq_errs

    def collect_t2(self):
        """Collect the T_2 of He, Ne, and Xe and return T2 dictionary"""
        T2 = {}
        keys = ['H', 'N', 'X']
        for label in self.det_labels:
            T2[label] = {}
            for k in keys:
                T2[label][k] = np.array([i[label]['T2'][k] for i in self.shot_data])
        return T2

    def collect_amps(self):
        """Collect all amplitudes with different frequencies and return amps dictionary"""
        amps = {}
        new_fkeys = list(set(self.fkeys) - set(['CP']))
        for label in self.det_labels:
            amps[label] = {}
            for f in new_fkeys:
                amps[label][f] = []
                for i in range(len(self.shot_data)):
                    amps[label][f].append(self.shot_data[i][label]['amps'][f])
                amps[label][f] = np.array(amps[label][f])
        return amps

    def collect_residuals(self):
        """Collect all block phase residuals with different frequencies and return phases_res dictionary"""
        phase_res = {}
        for label in self.det_labels:
            phase_res[label] = {}
            for f in self.fkeys:
                phase_res[label][f] = []
                for i in range(len(self.shot_data)):
                    phase_res[label][f].append(self.shot_data[i][label]['residuals'][f])
                phase_res[label][f] = np.array(phase_res[label][f])
        return phase_res

    def collect_abs_res_max(self):
        abs_res_max = {}
        for label in self.det_labels:
            abs_res_max[label] = [self.shot_data[i][label]['abs_res_max'] for i in range(len(self.shot_data))]
        return abs_res_max

    def get_sequence_per_cycle(self):
        spc = []
        for i in range(self.cycle_total):
            index = list(map(lambda x: x == str(i), self.cycle_number))
            spc.append(np.array(self.sequence)[index])
        spc = np.array(spc)
        return spc

    def get_freqs_per_cycle(self):
        freqs_per_cycle = {}
        freq_errs_per_cycle = {}
        labels = self.det_labels + self.dark_labels
        for label in labels:
            freqs_per_cycle[label] = {}
            freq_errs_per_cycle[label] = {}
            for f in self.fkeys:
                freqs_per_cycle[label][f] = []
                freq_errs_per_cycle[label][f] = []
                for i in range(self.cycle_total):
                    index = list(map(lambda x: x == str(i), self.cycle_number))
                    freqs_per_cycle[label][f].append(self.freqs[label][f][index])
                    freq_errs_per_cycle[label][f].append(self.freq_errs[label][f][index])
                freqs_per_cycle[label][f] = np.array(freqs_per_cycle[label][f])
                freq_errs_per_cycle[label][f] = np.array(freq_errs_per_cycle[label][f])
        return freqs_per_cycle, freq_errs_per_cycle

    def get_amps_per_cycle(self):
        amps_per_cycle = {}
        new_fkeys = list(set(self.fkeys) - set(['CP']))
        for label in self.det_labels:
            amps_per_cycle[label] = {}
            for f in new_fkeys:
                amps_per_cycle[label][f] = []
                for i in range(self.cycle_total):
                    index = list(map(lambda x: x == str(i), self.cycle_number))
                    amps_per_cycle[label][f].append(self.amps[label][f][index])
        return amps_per_cycle

    def calc_chi_sqaure(self):
        fkeys = self.fkeys
        labels = self.det_labels + self.dark_labels
        freqs = self.freqs
        freq_errs = self.freq_errs
        chi_square = {}

        for l in labels:
            chi_square[l] = {}
            for f in fkeys:
                freq_mean = np.mean(freqs[l][f])
                chi_square[l][f] = np.sum((freqs[l][f] - freq_mean)**2/freq_errs[l][f]**2)/len(freqs[l][f])
        return chi_square

    def plot_abs_res_max(self):
        shot_number = self.shot_number_int
        abs_res_max = self.abs_res_max

        plt.figure()
        for label in self.det_labels:
            plt.plot(shot_number, abs_res_max[label], '-o')
        plt.legend(self.det_labels)
        plt.xlabel('Shot number')
        plt.ylabel('abs residual max')
        plt.show()

    def plot_t2(self):
        """Plot T2 of He, Ne, and Xe versus shot number"""
        shot_number = self.shot_number_int
        T2 = self.T2

        for label in self.det_labels:
            plt.figure(1)
            plt.plot(shot_number, T2[label]['X'], '-x')
            plt.xlabel('shot number')
            plt.ylabel('$T_2$ of Xe [s]')
            plt.title('Run ' + self.run_number)

            plt.figure(2)
            plt.plot(shot_number, T2[label]['N'], '-^')
            plt.xlabel('shot number')
            plt.ylabel('$T_2$ of Ne [s]')

            plt.figure(3)
            plt.plot(shot_number, T2[label]['H'], '-v')
            plt.xlabel('shot number')
            plt.ylabel('$T_2$ of He [s]')
        plt.show()

    def plot_freqs_shot(self):
        """Plot frequencies of CP, He, Ne and Xe with errorbars versus shot number"""
        shot_number = self.shot_number_int
        freqs = self.freqs
        freq_err = self.freq_errs

        fig, ax = plt.subplots(2, 2, sharex=True)
        fig.suptitle('Run ' + self.run_number + ' light')
        fig.tight_layout()
        plt.subplots_adjust(top=0.87)

        for label in self.det_labels:
            ax[0][0].errorbar(shot_number, freqs[label]['CP'], yerr=freq_err[label]['CP'], fmt='-o')
            ax[0][0].set_ylabel('$\omega_{CP}$')

        for label in self.det_labels:
            ax[0][1].errorbar(shot_number, freqs[label]['H'], yerr=freq_err[label]['H'], fmt='-v')
            ax[0][1].set_ylabel('$\omega_{He}$')

        for label in self.det_labels:
            ax[1][0].errorbar(shot_number, freqs[label]['N'], yerr=freq_err[label]['N'], fmt='-^')
            ax[1][0].set_xlabel('shot number')
            ax[1][0].set_ylabel('$\omega_{Ne}$')

        for label in self.det_labels:
            ax[1][1].errorbar(shot_number, freqs[label]['X'], yerr=freq_err[label]['X'], fmt='-x')
            ax[1][1].set_xlabel('shot number')
            ax[1][1].set_ylabel('$\omega_{Xe}$')

        fig2, ax2 = plt.subplots(2, 2, sharex=True)
        fig2.suptitle('Run ' + self.run_number + ' dark')
        fig2.tight_layout()
        plt.subplots_adjust(top=0.87)

        for label in self.dark_labels:
            ax2[0][0].errorbar(shot_number, freqs[label]['CP'], yerr=freq_err[label]['CP'], fmt='-o')
            ax2[0][0].set_ylabel('$\omega_{CP}$')

        for label in self.dark_labels:
            ax2[0][1].errorbar(shot_number, freqs[label]['H'], yerr=freq_err[label]['H'], fmt='-v')
            ax2[0][1].set_ylabel('$\omega_{He}$')

        for label in self.dark_labels:
            ax2[1][0].errorbar(shot_number, freqs[label]['N'], yerr=freq_err[label]['N'], fmt='-^')
            ax2[1][0].set_xlabel('shot number')
            ax2[1][0].set_ylabel('$\omega_{Ne}$')

        for label in self.dark_labels:
            ax2[1][1].errorbar(shot_number, freqs[label]['X'], yerr=freq_err[label]['X'], fmt='-x')
            ax2[1][1].set_xlabel('shot number')
            ax2[1][1].set_ylabel('$\omega_{Xe}$')

        plt.show()
        return fig, ax

    def plot_amps_shot(self, a, b):
        """Plot mean value of amplitudes in (a, b) of He, Ne and Xe versus shot number """
        shot_number = self.shot_number_int
        amps = self.amps

        he_amps = {}
        ne_amps = {}
        xe_amps = {}
        for label in self.det_labels:
            he_amps[label] = [np.mean(amps[label]['H'][i][a:b]) for i in range(len(self.shot_number))]
            ne_amps[label] = [np.mean(amps[label]['N'][i][a:b]) for i in range(len(self.shot_number))]
            xe_amps[label] = [np.mean(amps[label]['X'][i][a:b]) for i in range(len(self.shot_number))]

        for label in self.det_labels:
            plt.figure()
            plt.title('Run ' + self.run_number + ' detection ' + label)
            plt.plot(shot_number, he_amps[label], '-v', shot_number, ne_amps[label], '-^', shot_number, xe_amps[label],
                     '-x')
            plt.legend(['He', 'Ne', 'Xe'])
            plt.xlabel('shot number')
            plt.ylabel('Amplitude [V]')
        plt.show()

    def plot_freqs_sequence(self):
        """Plot frequencies of CP, He, Ne and Xe with errorbars versus Ne rotation angle"""
        sequence_per_cycle = self.sequence_per_cycle
        freqs_per_cycle = self.freqs_per_cycle
        freq_errs_per_cycle = self.freq_errs_per_cycle

        for label in self.det_labels:
            fig, ax = plt.subplots(2, 2, sharex=True)
            fig.suptitle('Run ' + self.run_number + ' detection ' + label)
            fig.tight_layout()
            plt.subplots_adjust(top=0.87)
            for i in range(self.cycle_total):
                ax[0][0].errorbar(sequence_per_cycle[i], freqs_per_cycle[label]['CP'][i], yerr=freq_errs_per_cycle[label]['CP'][i],
                                  fmt='-o')
            ax[0][0].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
            ax[0][0].set_ylabel('$\omega_{CP}$')

            for i in range(self.cycle_total):
                ax[0][1].errorbar(sequence_per_cycle[i], freqs_per_cycle[label]['H'][i], yerr=freq_errs_per_cycle[label]['H'][i],
                                  fmt='-o')
            ax[0][1].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
            ax[0][1].set_ylabel('$\omega_{He}$')

            for i in range(self.cycle_total):
                ax[1][0].errorbar(sequence_per_cycle[i], freqs_per_cycle[label]['N'][i], yerr=freq_errs_per_cycle[label]['N'][i],
                                  fmt='-o')
            ax[1][0].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
            ax[1][0].set_xlabel(self.sequence_name)
            ax[1][0].set_ylabel('$\omega_{Ne}$')

            for i in range(self.cycle_total):
                ax[1][1].errorbar(sequence_per_cycle[i], freqs_per_cycle[label]['X'][i], yerr=freq_errs_per_cycle[label]['X'][i],
                                  fmt='-o')
            ax[1][1].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
            ax[1][1].set_xlabel(self.sequence_name)
            ax[1][1].set_ylabel('$\omega_{Xe}$')

        fig, ax = plt.subplots(2, 2, sharex=True)
        fig.suptitle('Run ' + self.run_number + ' dark')
        fig.tight_layout()
        plt.subplots_adjust(top=0.87)
        for label in self.dark_labels:
            for i in range(self.cycle_total):
                ax[0][0].errorbar(sequence_per_cycle[i], freqs_per_cycle[label]['CP'][i],
                                  yerr=freq_errs_per_cycle[label]['CP'][i],
                                  fmt='-o')
            ax[0][0].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
            ax[0][0].set_ylabel('$\omega_{CP}$')

            for i in range(self.cycle_total):
                ax[0][1].errorbar(sequence_per_cycle[i], freqs_per_cycle[label]['H'][i],
                                  yerr=freq_errs_per_cycle[label]['H'][i],
                                  fmt='-o')
            ax[0][1].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
            ax[0][1].set_ylabel('$\omega_{He}$')

            for i in range(self.cycle_total):
                ax[1][0].errorbar(sequence_per_cycle[i], freqs_per_cycle[label]['N'][i],
                                  yerr=freq_errs_per_cycle[label]['N'][i],
                                  fmt='-o')
            ax[1][0].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
            ax[1][0].set_xlabel(self.sequence_name)
            ax[1][0].set_ylabel('$\omega_{Ne}$')

            for i in range(self.cycle_total):
                ax[1][1].errorbar(sequence_per_cycle[i], freqs_per_cycle[label]['X'][i],
                                  yerr=freq_errs_per_cycle[label]['X'][i],
                                  fmt='-o')
            ax[1][1].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
            ax[1][1].set_xlabel(self.sequence_name)
            ax[1][1].set_ylabel('$\omega_{Xe}$')
        plt.show()
        return fig, ax

    def plot_amps_sequence(self, lb, ub):
        """Plot mean value of amplitudes in (a, b) of He, Ne and Xe versus shot number """
        amps_per_cycle = self.amps_per_cycle
        sequence_per_cycle = self.sequence_per_cycle

        fkey = ['H', 'N', 'X']
        mean_amps = {}
        for label in self.det_labels:
            mean_amps[label] = {}
            for f in fkey:
                mean_amps[label][f] = np.mean(np.array(amps_per_cycle[label][f])[:, :, lb:ub], 2)

        for label in self.det_labels:
            plt.figure()
            for i in range(self.cycle_total):
                plt.plot(sequence_per_cycle[i], mean_amps[label]['N'][i], '-^')
            plt.legend([str(i) for i in range(self.cycle_total)])
            plt.xlabel(self.sequence_name)
            plt.ylabel('Ne amplitude [V]')
            plt.title('Run ' + self.run_number + ' detection ' + label)

            plt.figure()
            for i in range(self.cycle_total):
                plt.plot(sequence_per_cycle[i], mean_amps[label]['H'][i], '-v')
            plt.legend([str(i) for i in range(self.cycle_total)])
            plt.xlabel(self.sequence_name)
            plt.ylabel('He amplitude [V]')

            plt.figure()
            for i in range(self.cycle_total):
                plt.plot(sequence_per_cycle[i], mean_amps[label]['X'][i], '-x')
            plt.legend([str(i) for i in range(self.cycle_total)])
            plt.xlabel(self.sequence_name)
            plt.ylabel('Xe amplitude [V]')

        plt.show()
        return mean_amps
