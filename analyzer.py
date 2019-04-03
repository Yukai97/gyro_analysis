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

    def __init__(self, run_number):
        self.run_number = run_number
        self.shotdir_run = os.path.join(shotdir, self.run_number)
        self.file_list = sorted(os.listdir(self.shotdir_run))
        self.shot_data = self.load_shot_data()
        self.timestamp = [i['shotinfo']['timestamp'] for i in self.shot_data]
        self.flag = (self.shot_data[0]['shotinfo']['exists'] == 'True')
        self.fkeys = self.shot_data[0]['fkeys']
        self.bl = self.shot_data[0]['block length']
        [freqs, freq_err] = self.collect_freqs_freqerr()
        self.freqs = freqs
        self.freq_err = freq_err
        self.T2 = self.collect_t2()
        if self.flag:
            self.shot_number = [i['shotinfo']['shot_number'] for i in self.shot_data]
            self.shot_number_int = list(map(int, self.shot_number))
            self.waveform_config = [i['shotinfo']['waveform_config'] for i in self.shot_data]
            self.he_angle = [i['shotinfo']['he_angle'] for i in self.shot_data]
            self.ne_angle = [i['shotinfo']['ne_angle'] for i in self.shot_data]
            self.xe_angle = [i['shotinfo']['xe_angle'] for i in self.shot_data]
        else:
            self.shot_number = [i.split('.')[0].split('_')[1] for i in self.file_list]
            self.shot_number_int = list(map(int, self.shot_number))
            self.he_angle = None
            self.ne_angle = None
            self.xe_angle = None
        self.amps = self.collect_amps()
        self.phase_res = self.collect_phase_res()

    def load_shot_data(self):
        """Load all shot data files in the same run specified by run_number and return a shot_data list"""
        file_list = self.file_list
        shot_data = []
        for i in range(len(file_list)):
            file_path = os.path.join(self.shotdir_run, file_list[i])
            with open(file_path, 'r') as read_data:
                print('loading file:' + file_path)
                shot_data.append(json.load(read_data))
        return shot_data

    def collect_freqs_freqerr(self):
        """Collect all frequencies and corresponding errorbars and return freqs and freq_err dictionaries"""
        freqs = {}
        freq_err = {}
        for f in self.fkeys:
            freqs[f] = np.array([i['freqs'][f] for i in self.shot_data])
            freq_err[f] = np.array([i['freq_err'][f] for i in self.shot_data])
        return freqs, freq_err

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
                phase_res[f].append(self.shot_data[i]['phase_res'][f])
            phase_res[f] = np.array(phase_res[f])
        return phase_res

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
        plt.title('Run' + self.run_number)

        plt.figure(3)
        plt.plot(shot_number, T2['H'], '-v')
        plt.xlabel('shot number')
        plt.ylabel('$T_2$ of He [s]')
        plt.title('Run' + self.run_number)
        plt.show()

    def plot_freqs_shot(self):
        """Plot frequencies of CP, He, Ne and Xe with errorbars versus shot number"""
        shot_number = self.shot_number_int
        freqs = self.freqs
        freq_err = self.freq_err

        plt.figure()
        plt.errorbar(shot_number, freqs['CP'], yerr=freq_err['CP'], fmt='-o')
        plt.xlabel('shot number')
        plt.ylabel('$\omega_{CP}$')
        plt.title('Run' + self.run_number)

        plt.figure()
        plt.errorbar(shot_number, freqs['H'], yerr=freq_err['H'], fmt='-v')
        plt.xlabel('shot number')
        plt.ylabel('$\omega_{He}$')
        plt.title('Run' + self.run_number)

        plt.figure()
        plt.errorbar(shot_number, freqs['N'], yerr=freq_err['N'], fmt='-^')
        plt.xlabel('shot number')
        plt.ylabel('$\omega_{Ne}$')
        plt.title('Run' + self.run_number)

        plt.figure()
        plt.errorbar(shot_number, freqs['X'], yerr=freq_err['X'], fmt='-x')
        plt.xlabel('shot number')
        plt.ylabel('$\omega_{Xe}$')
        plt.title('Run' + self.run_number)
        plt.show()

    def plot_freqs_angle(self):
        """Plot frequencies of CP, He, Ne and Xe with errorbars versus Ne rotation angle"""
        ne_angle = self.ne_angle
        freqs = self.freqs
        freq_err = self.freq_err

        plt.figure()
        plt.errorbar(ne_angle, freqs['CP'], yerr=freq_err['CP'], fmt='-o')
        plt.xlabel('Ne angle')
        plt.ylabel('$\omega_{CP}$')
        plt.title('Run' + self.run_number)

        plt.figure()
        plt.errorbar(ne_angle, freqs['H'], yerr=freq_err['H'], fmt='-v')
        plt.xlabel('Ne angle')
        plt.ylabel('$\omega_{He}$')
        plt.title('Run' + self.run_number)

        plt.figure()
        plt.errorbar(ne_angle, freqs['N'], yerr=freq_err['N'], fmt='-^')
        plt.xlabel('Ne angle')
        plt.ylabel('$\omega_{Ne}$')
        plt.title('Run' + self.run_number)

        plt.figure()
        plt.errorbar(ne_angle, freqs['X'], yerr=freq_err['X'], fmt='-x')
        plt.xlabel('Ne angle')
        plt.ylabel('$\omega_{Xe}$')
        plt.title('Run' + self.run_number)
        plt.show()

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
