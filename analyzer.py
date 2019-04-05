import numpy as np
from matplotlib import pyplot as plt
import os
import json
from gyro_analysis.local_path import shotdir


class RunAnalyzer:
    """Analyze all shot data files in the same run specified by the run_number and plot related figures

    Args:
        run_number: specify the run we need to analyze
    """

    def __init__(self, run_number, sequence_name):
        self.run_number = run_number
        self.sequence_name = str(sequence_name)
        self.shotdir_run = os.path.join(shotdir, self.run_number)
        self.file_list = sorted(os.listdir(self.shotdir_run))
        self.shot_data = self.load_shot_data()
        self.timestamp = [i['shotinfo']['timestamp'] for i in self.shot_data]
        self.flag = (self.shot_data[0]['shotinfo']['exists'] == True)
        self.fkeys = self.shot_data[0]['fkeys']
        self.bl = self.shot_data[0]['block length']
        [freqs, freq_err] = self.collect_freqs_freqerr()
        self.freqs = freqs
        self.freq_errs = freq_err
        self.T2 = self.collect_t2()
        self.amps = self.collect_amps()
        self.phase_res = self.collect_phase_res()
        if self.flag:
            self.shot_number = [i['shotinfo']['shot_number'] for i in self.shot_data]
            self.shot_number_int = list(map(int, self.shot_number))
            self.waveform_config = [i['shotinfo']['waveform_config'] for i in self.shot_data]
            self.he_angle = [i['shotinfo']['he_angle'] for i in self.shot_data]
            self.ne_angle = [i['shotinfo']['ne_angle'] for i in self.shot_data]
            self.xe_angle = [i['shotinfo']['xe_angle'] for i in self.shot_data]
            self.cycle_number = [i['shotinfo']['cycle_number'] for i in self.shot_data]
            self.cycle_total = int(max(self.cycle_number)) + 1
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
                shot_data.append(json.load(read_data))
        return shot_data

    def collect_freqs_freqerr(self):
        """Collect all frequencies and corresponding errorbars and return freqs and freq_err dictionaries"""
        freqs = {}
        freq_errs = {}
        for f in self.fkeys:
            freqs[f] = np.array([i['freqs'][f] for i in self.shot_data])
            freq_errs[f] = np.array([i['freq_err'][f] for i in self.shot_data])
        return freqs, freq_errs

    def collect_t2(self):
        """Collect the T_2 of He, Ne, and Xe and return T2 dictionary"""
        T2 = {}
        keys = ['H', 'N', 'X']
        for k in keys:
            T2[k] = np.array([i['T2'][k] for i in self.shot_data])
        return T2

    def collect_amps(self):
        """Collect all amplitudes with different frequencies and return amps dictionary"""
        amps = {}
        new_fkeys = list(set(self.fkeys) - set(['CP']))
        for f in new_fkeys:
            amps[f] = []
            for i in range(len(self.shot_data)):
                amps[f].append(self.shot_data[i]['amps'][f])
            amps[f] = np.array(amps[f])
        return amps

    def collect_phase_res(self):
        """Collect all block phase residuals with different frequencies and return phases_res dictionary"""
        phase_res = {}
        for f in self.fkeys:
            phase_res[f] = []
            for i in range(len(self.shot_data)):
                phase_res[f].append(self.shot_data[i]['residuals'][f])
            phase_res[f] = np.array(phase_res[f])
        return phase_res

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
        for f in self.fkeys:
            freqs_per_cycle[f] = []
            freq_errs_per_cycle[f] = []
            for i in range(self.cycle_total):
                index = list(map(lambda x: x == str(i), self.cycle_number))
                freqs_per_cycle[f].append(self.freqs[f][index])
                freq_errs_per_cycle[f].append(self.freq_errs[f][index])
            freqs_per_cycle[f] = np.array(freqs_per_cycle[f])
            freq_errs_per_cycle[f] = np.array(freq_errs_per_cycle[f])
        return freqs_per_cycle, freq_errs_per_cycle

    def get_amps_per_cycle(self):
        amps_per_cycle = {}
        new_fkeys = list(set(self.fkeys) - set(['CP']))
        for f in new_fkeys:
            amps_per_cycle[f] = []
            for i in range(self.cycle_total):
                index = list(map(lambda x: x == str(i), self.cycle_number))
                amps_per_cycle[f].append(self.amps[f][index])
        return amps_per_cycle

    def plot_t2(self):
        """Plot T2 of He, Ne, and Xe versus shot number"""
        shot_number = self.shot_number_int
        T2 = self.T2

        plt.figure(1)
        plt.plot(shot_number, T2['X'], '-x')
        plt.xlabel('shot number')
        plt.ylabel('$T_2$ of Xe [s]')
        plt.title('Run' + self.run_number)

        plt.figure(2)
        plt.plot(shot_number, T2['N'], '-^')
        plt.xlabel('shot number')
        plt.ylabel('$T_2$ of Ne [s]')

        plt.figure(3)
        plt.plot(shot_number, T2['H'], '-v')
        plt.xlabel('shot number')
        plt.ylabel('$T_2$ of He [s]')
        plt.show()

    def plot_freqs_shot(self):
        """Plot frequencies of CP, He, Ne and Xe with errorbars versus shot number"""
        shot_number = self.shot_number_int
        freqs = self.freqs
        freq_err = self.freq_errs

        fig, ax = plt.subplots(2, 2, sharex=True)
        fig.suptitle('Run ' + self.run_number)
        fig.tight_layout()
        plt.subplots_adjust(top=0.87)

        ax[0][0].errorbar(shot_number, freqs['CP'], yerr=freq_err['CP'], fmt='-o')
        ax[0][0].set_ylabel('$\omega_{CP}$')

        ax[0][1].errorbar(shot_number, freqs['H'], yerr=freq_err['H'], fmt='-v')
        ax[0][1].set_ylabel('$\omega_{He}$')

        ax[1][0].errorbar(shot_number, freqs['N'], yerr=freq_err['N'], fmt='-^')
        ax[1][0].set_xlabel('shot number')
        ax[1][0].set_ylabel('$\omega_{Ne}$')

        ax[1][1].errorbar(shot_number, freqs['X'], yerr=freq_err['X'], fmt='-x')
        ax[1][1].set_xlabel('shot number')
        ax[1][1].set_ylabel('$\omega_{Xe}$')
        plt.show()
        return fig, ax

    def plot_amps_shot(self, a, b):
        """Plot mean value of amplitudes in (a, b) of He, Ne and Xe versus shot number """
        shot_number = self.shot_number_int
        amps = self.amps

        he_amps = [np.mean(amps['H'][i][a:b]) for i in self.shot_number_int]
        ne_amps = [np.mean(amps['N'][i][a:b]) for i in self.shot_number_int]
        xe_amps = [np.mean(amps['X'][i][a:b]) for i in self.shot_number_int]

        plt.figure()
        plt.plot(shot_number, he_amps, '-v', shot_number, ne_amps, '-^', shot_number, xe_amps, '-x')
        plt.legend(['He', 'Ne', 'Xe'])
        plt.xlabel('shot number')
        plt.ylabel('Amplitude [V]')
        plt.show()

    def plot_freqs_sequence(self):
        """Plot frequencies of CP, He, Ne and Xe with errorbars versus Ne rotation angle"""
        sequence = np.array(self.sequence_per_cycle).transpose()
        freqs_per_cycle = {}
        for f in self.fkeys:
            freqs_per_cycle[f] = np.array(self.freqs_per_cycle[f]).transpose()

        fig, ax = plt.subplots(2, 2, sharex=True)
        fig.suptitle('Run ' + self.run_number)
        fig.tight_layout()
        plt.subplots_adjust(top=0.87)

        ax[0][0].plot(sequence, freqs_per_cycle['CP'], '-o')
        ax[0][0].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
        ax[0][0].set_ylabel('$\omega_{CP}$')

        ax[0][1].plot(sequence, freqs_per_cycle['H'], '-v')
        ax[0][1].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
        ax[0][1].set_ylabel('$\omega_{He}$')

        ax[1][0].plot(sequence, freqs_per_cycle['N'], '-^')
        ax[1][0].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
        ax[1][0].set_xlabel(self.sequence_name)
        ax[1][0].set_ylabel('$\omega_{Ne}$')

        ax[1][1].plot(sequence, freqs_per_cycle['X'], '-x')
        ax[1][1].legend([str(i) for i in range(self.cycle_total)], fontsize=10)
        ax[1][1].set_xlabel(self.sequence_name)
        ax[1][1].set_ylabel('$\omega_{Xe}$')

        plt.show()
        return fig, ax

    def plot_amps_sequence(self, a, b):
        """Plot mean value of amplitudes in (a, b) of He, Ne and Xe versus shot number """
        amps_per_cycle = self.amps_per_cycle

        fkey = ['H', 'N', 'X']
        mean_amps = {}
        for f in fkey:
            mean_amps[f] = np.mean(np.array(amps_per_cycle[f])[:, :, a:b], 2)
            mean_amps[f] = mean_amps[f].transpose()
        sequence = np.array(self.sequence_per_cycle).transpose()

        plt.figure()
        plt.plot(sequence, mean_amps['N'], '-^')
        plt.legend([str(i) for i in range(self.cycle_total)])
        plt.xlabel(self.sequence_name)
        plt.ylabel('Ne amplitude [V]')

        plt.figure()
        plt.plot(sequence, mean_amps['H'], '-v')
        plt.legend([str(i) for i in range(self.cycle_total)])
        plt.xlabel(self.sequence_name)
        plt.ylabel('He amplitude [V]')

        plt.figure()
        plt.plot(sequence, mean_amps['X'], '-x')
        plt.legend([str(i) for i in range(self.cycle_total)])
        plt.xlabel(self.sequence_name)
        plt.ylabel('Xe amplitude [V]')

        plt.show()

