import numpy as np
from numpy import sin
from numpy import cos
from scipy import signal
from scipy import fftpack
from scipy import optimize as opt
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import json_tricks as jsont
from gyro_analysis.shotinfo import ShotInfo
import gyro_analysis.local_path as lp

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.unicode_minus'] = False
mpl.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.formatter.limits'] = (-2, 2)
# mpl.rc('text', usetex=True)

HENE_RATIO = 9.650

####
####
#### '{:04d}'.format change number to int(number)
####
#### load_info needs run_number not name (get rid of all names)
####


class SClist:
    """
    Stores and manipulates a list of SCdat objects plus a header.

    SClist.l contains the list of SCdat o917-957-3766bjects one for each block
    SClist.fkeys  contains a list of the species whose signals are being computed.
    SClist.amps, contains amplitudes for each signal
    SClist.total_phases contains phases for each signal
    SClist.fp (or freqs_and_phases) contains dict with keys:
        'phase_start', 'phase_start_err', 'phase_end', 'phase_end_err'
        'freq_start', 'freq_start_err', 'freq_end', 'freq_end_err'
        Each value is a dict with frequencies.
    SClist.res contains residuals 'res_start', 'res_end' for fits done with t_start = 0 or t_end = 0.


    """

    def __init__(self, run_number, shot_number, label):
        self.ext = '.scf'
        self.label = label
        self.run_number = '{:04d}'.format(int(run_number))
        self.shot_number = '{:03d}'.format(int(shot_number))
        self.file_name = '_'.join([self.run_number, self.shot_number])+self.label
        self.full_name = os.path.join(lp.shotdir)
        self.hene_ratio = HENE_RATIO
        self.ext = '.scf'
        self.shotinfo = ShotInfo(run_number, shot_number)
        self.scdir = lp.scdir
        self.path = {'homedir': lp.homedir, 'rawdir': lp.rawdir,
                     'infodir': lp.infodir, 'scdir': lp.scdir, 'shotdir': lp.shotdir}
        self.write_dir = os.path.join(self.path['shotdir'], self.run_number)
        self.fullname = os.path.join(lp.scdir, self.run_number, self.file_name + self.ext)
        with open(self.fullname, 'r') as read_data:
            data = jsont.load(read_data)
        self.hdr = data[-1]
        self.l = data[:-1]
        self.keys = list(self.l[0].keys())
        self.fkeys = list(self.l[0]['w'].keys())
        self.dt = self.l[0]['dt']
        self.amp_phase()
        self.time = np.array([i['start'] * self.dt for i in self.l])
        self.end_time = self.l[-1]['end'] * self.dt
        self.amps = self.collect_amps()
        self.block_phases = self.collect_block_phases()

        self.bl = (self.l[0]['end'] - self.l[0]['start']) * self.dt
        self.total_phases = self.collect_total_phases()
        self.ne_corr = self.corrected_phases()
        self.T2 = self.fit_T2()
        self.fkeys.append('CP')

        freqs_and_phases, residuals = self.fit_phases()
        self.freqs_and_phases = freqs_and_phases
        self.fp = freqs_and_phases
        self.res_dict = residuals
        self.freq = self.fp['freq_start']
        self.freq_err = self.fp['freq_start_err']
        self.phase_start = self.fp['phase_start']
        self.phase_end = self.fp['phase_end']
        self.phase_err = self.fp['phase_start_err']
        self.residuals = self.res_dict['res_start']

    def amp_phase(self):
        """
        add 'amp' and 'bp' dicts to each block object in SClist.l;
        keys are the names of the signal frequencies;
        values are the amplitude ('amp') or local phase ('bp') of that block

        :return:
        """
        for el in self.l:
            el['amp'] = {}
            el['bp'] = {}
            for f in self.fkeys:
                a = el['a'][f]
                b = el['b'][f]
                el['amp'][f] = np.sqrt(a ** 2 + b ** 2)
                el['bp'][f] = np.arctan2(b, a)
        return

    def collect_block_phases(self):
        """
        make dictionary of {signal: block phase array} for each signal

        :return: dictionary
        """
        pdict = {}
        for f in self.fkeys:
            pdict[f] = [el['bp'][f] for el in self.l]
        return pdict

    def collect_amps(self):
        """
        make dictionary of {signal: amplitudes array} for each signal

        :return: dictionary
        """
        adict = {}
        for f in self.fkeys:
            adict[f] = [el['amp'][f] for el in self.l]
        return adict

    def collect_total_phases(self):
        """
        make dictionary of {signal: total phases} for each signal

        :return:  dict
        """
        l = self.l
        for el in l:
            el['tp'] = {}
        tot_ph_dict = {}
        for fk in self.fkeys:
            freq = l[0]['w'][fk]
            phase_per_block = freq * self.bl
            block_ph = self.block_phases[fk]
            total_ph = np.zeros_like(block_ph)
            total_ph[0] = block_ph[0]
            l[0]['tp'][fk] = total_ph[0]
            for j in range(1, len(block_ph)):
                guess = total_ph[j - 1] + phase_per_block
                total_ph[j] = np.unwrap([guess, block_ph[j]])[1]
                l[j]['tp'][fk] = total_ph[j]  # save to self.l as well
            tot_ph_dict[fk] = total_ph
        return tot_ph_dict

    def corrected_phases(self):
        """
        add neon corrected frequency to the 'tp' dict inside each block object of SClist.l

        :return:
        """
        gr = 1 / self.hene_ratio
        ne_ph = self.total_phases['N']
        he_ph = self.total_phases['H']
        corr_ne_ph = ne_ph - gr * he_ph
        self.total_phases['CP'] = corr_ne_ph
        for i in range(len(self.l)):
            self.l[i]['tp']['CP'] = corr_ne_ph[i]
        return corr_ne_ph

    def fit_T2(self):
        """get T2 of He, Ne and Xe"""
        def exp_decay(t, a, b):
            return a * np.exp(-t * b)

        atoms = ['H', 'N', 'X']
        initial_guess = {'H': [0.5, 0.0002], 'N': [0.8, 0.0001], 'X': [0.015, 0.02]}
        T2 = {}
        for i in atoms:
            amps = self.amps[i]
            t = self.time
            popt, pcov = opt.curve_fit(exp_decay, t, amps, initial_guess[i])
            T2[i] = 1/popt[1]
        return T2

    def fit_phases(self):
        """Use linear regression to fit phases to a line (frequency), start phase and end phase.

        Need two fits one for time t=0 at start, the other for time t0 at end, so that the phase
        error is minimized at both the end and beginning of data.
        
        variables ending in '1' are part of fit with t = 0,...,tn
        variables ending in '2' are part of fit with t = -tn,...,0

        :return: dictionary with frequencies (start/end), phases (start/end), errors and residuals
        """

        phase_start = {}
        phase_start_err = {}
        phase_end = {}
        phase_end_err = {}
        freq_start = {}
        freq_start_err = {}
        freq_end = {}
        freq_end_err = {}
        res_start = {}
        res_end = {}

        t1 = self.time
        t2 = self.time-self.time[-1]  # renumber so last point is 0 to find phase at the end of the section
        bf1 = np.array([np.ones_like(t1), t1]).transpose()
        bf2 = np.array([np.ones_like(t2), t2]).transpose()
        for f in self.fkeys:
            phases = self.total_phases[f]
            lf1 = opt.lsq_linear(bf1, phases, tol=1e-16)
            lf2 = opt.lsq_linear(bf2, phases, tol=1e-16)
            if not lf1['success'] or not lf2['success']:
                raise RuntimeError('Fit of phases to line did not converge')
            r1 = lf1['fun']  # residual
            r2 = lf2['fun']
            err1 = self.calc_errors(r1, t1)
            err2 = self.calc_errors(r2, t2)

            phase_start[f] = lf1['x'][0]  # phase
            phase_start_err[f] = err1[0]
            phase_end[f] = lf2['x'][0]
            phase_end_err[f] = err2[0]
            freq_start[f] = lf1['x'][1]  # frequency
            freq_start_err[f] = err1[1]
            freq_end[f] = lf2['x'][1]
            freq_end_err[f] = err2[1]
            res_start[f] = r1
            res_end[f] = r2
        return [dict(phase_start=phase_start, phase_start_err=phase_start_err,
                    phase_end=phase_end, phase_end_err=phase_start_err,
                    freq_start=freq_start, freq_start_err=freq_start_err,
                    freq_end=freq_start, freq_end_err=freq_start_err),
                    dict(res_start=res_start, res_end=res_end)]

    def calc_errors(self, res, times):
        """ Compute errors on linear regression parameters from residuals """

        t = times
        ta = np.mean(t)
        tq = np.sum((t-ta)**2)
        n = len(t)
        slope_err = np.sqrt(np.sum(res ** 2) / (n - 2) / tq)
        intercept_err = slope_err * np.sqrt(np.sum(t ** 2) / n)
        return intercept_err, slope_err

    def plot_amp(self, fk):
        """

        :param fk: frequency key to plot
        :return:
        """
        fig, ax = plt.subplots()
        ax.plot(self.time, [i['amp'][fk] for i in self.l])
        ax.set_ylabel('{} Amplitude [V]'.format(fk + 'e'))
        ax.set_xlabel('Time [s]')
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.17)
        ax.set_title('Run {}, block length {}'.format(self.file_name, self.bl))
        return fig, ax

    def plot_phase(self, fk):
        fig, ax = plt.subplots()
        ax.plot(self.time, [i['tp'][fk] for i in self.l])
        ax.set_ylabel('{} Amplitude [V]'.format(fk + 'e'))
        ax.set_xlabel('Time [s]')
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.17)
        ax.set_title('Run {}, block length {}'.format(self.file_name, self.bl))
        return fig, ax

    def plot_phase_res(self):
        """ Plot Ne, He, corrected residuals after linear fit """

        r = self.phase_res
        t = self.time
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(t, r['N'])
        ax[0].set_ylabel('Ne')
        # ax[0].set_ylabel('Ne phase \nres. [rad]')
        ax[1].plot(t, r['H'])
        ax[1].set_ylabel('He')
        # ax[1].set_ylabel('He phase \nres. [rad]')
        ax[2].plot(t, r['CP'])
        ax[2].set_ylabel('Ne (B-inv)')
        # ax[2].set_ylabel('B-inv. phase \nres. (Ne) [rad]')
        big_ax = fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        big_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        big_ax.set_xlabel("time [s]")
        plt.subplots_adjust(left=0.2)
        by = big_ax.yaxis
        by.set_label_text("phase residual [rad]")
        by.set_label_coords(-0.2, 0.5)
        return fig, ax, big_ax

    def plot_phase_res_psd(self, f, a, b):
        rs = self.phase_res[f]
        n = len(rs)
        t = self.bl
        x = np.linspace(0.0, n * t, n)
        r = np.sin(a * 2.0 * np.pi * x) + 0.5 * np.sin(b * 2.0 * np.pi * x)
        win = signal.hamming(n)
        rf = fftpack.fft(r*win)
        xf = np.linspace(0, 1.0/2/t, n/2)
        fig, ax = plt.subplots()
        ax.semilogy(xf[1:n//2], 2*np.abs(rf[1:n//2])/n)
        return fig, ax

    def write_json(self, l=''):
        """
        Write json with some relevant outputs of analyzing the phase data

        """

        file_name = self.file_name + l
        if not os.path.isdir(self.write_dir):
            os.makedirs(self.write_dir)
        file_path = os.path.join(self.write_dir, file_name + '.shd')

        def default_json(o):
            return o.__dict__

        f = open(file_path, 'w')
        output_dict = {'fkeys': self.fkeys, 'freq': self.freq,
                       'freq_err': self.freq_err, 'residuals': self.residuals,
                       'T2': self.T2, 'amps': self.amps, 'phase_start': self.phase_start,
                       'phase_end': self.phase_end, 'phase_err': self.phase_err,
                       'block length': self.bl}
        json_output = jsont.dumps(output_dict)
        f.write(json_output)
        f.close()
        return


class ShdReader:
    """ Class for reading .sco files output by SClist """

    def __init__(self, run_number, shot_number, label):
        self.ext = '.shd'
        self.label = label
        self.run_number = '{:04d}'.format(int(run_number))
        self.shot_number = '{:03d}'.format(int(shot_number))
        self.file_name = '_'.join([self.run_number, self.shot_number])+label+self.ext
        self.full_name = os.path.join(lp.shotdir, self.run_number, self.file_name)
        with open(self.full_name, 'r') as read_data:
            scdata = jsont.load(read_data)
        self.fkeys = scdata['fkeys']
        self.freq = scdata['freq']
        self.freq_err = scdata['freq_err']
        self.residuals = scdata['residuals']
        self.T2 = scdata['T2']
        self.amps = scdata['amps']
        self.phase_start = scdata['phase_start']
        self.phase_end = scdata['phase_end']
        self.phase_err = scdata['phase_err']
        self.bl = scdata['block length']
        self.shotinfo = ShotInfo(self.run_number, self.shot_number)
        self.si = self.shotinfo
