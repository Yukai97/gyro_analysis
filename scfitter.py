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
from gyro_analysis.local_path import paths as lp
from gyro_analysis.local_path import extensions as ext

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
    Stores and manipulates a list of RawBlock objects plus a header.

    SClist.block_data contains the list of RawBlock objects one for each block
    SClist.fkeys  contains a list of the species whose signals are being computed.
    SClist.amps, contains amplitudes for each signal
    SClist.total_phases contains phases for each signal
    SClist.fp (or freqs_and_phases) contains dict with keys:
        'phase_start', 'phase_start_err', 'phase_end', 'phase_end_err'
        'freq_start', 'freq_start_err', 'freq_end', 'freq_end_err'
        Each value is a dict with frequencies.
    SClist.res contains residuals 'res_start', 'res_end' for fits done with t_start = 0 or t_end = 0.


    """

    def __init__(self, run_number, shot_number, label, phase_offset={}, drop_blocks=[]):
        self.label = label
        self.run_number = '{:04d}'.format(int(run_number))
        self.shot_number = '{:03d}'.format(int(shot_number))
        self.file_name = '_'.join([self.run_number, self.shot_number])+self.label
        self.phase_offset = phase_offset   # Expected initial phase for Helium and Neon.  Needed when analyzing dark times
        self.hene_ratio = HENE_RATIO
        self.shotinfo = ShotInfo(run_number, shot_number)
        self.outdir = os.path.join(lp.scodir, self.run_number)
        self.fullname = os.path.join(lp.rfodir, self.run_number, self.file_name + ext.sc_in)
        with open(self.fullname, 'r') as read_data:
            data = jsont.load(read_data)
        self.hdr = data[-1]
        self.block_data = data[:-1]
        self.block_index = np.array(list(range(len(self.block_data))))
        self.drop_blocks = drop_blocks
        self.block_mask = np.invert(np.isin(self.block_index, self.drop_blocks))
        self.keys = list(self.block_data[0].keys())
        self.fkeys = list(self.block_data[0]['w'].keys())
        self.dt = self.block_data[0]['dt']
        self.amp_phase()
        self.time = np.array([i['start'] * self.dt for i in self.block_data])
        self.end_time = self.block_data[-1]['end'] * self.dt
        self.amps = self.collect_amps()
        self.block_phases = self.collect_block_phases()

        self.bl = (self.block_data[0]['end'] - self.block_data[0]['start']) * self.dt
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
        self.n_blocks = len(self.residuals[self.fkeys[0]])
        self.phase_end_time = self.bl * (self.n_blocks - 1)  # time of phase_end point.

    def amp_phase(self):
        """
        add 'amp' and 'bp' dicts to each block object in SClist.l;
        keys are the names of the signal frequencies;
        values are the amplitude ('amp') or local phase ('bp') of that block

        :return:
        """
        for el in self.block_data:
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
            pdict[f] = [el['bp'][f] for el in self.block_data]
        return pdict

    def collect_amps(self):
        """
        make dictionary of {signal: amplitudes array} for each signal

        :return: dictionary
        """
        adict = {}
        for f in self.fkeys:
            adict[f] = [el['amp'][f] for el in self.block_data]
        return adict

    def collect_total_phases(self):
        """
        make dictionary of {signal: total phases} for each signal

        :return:  dict
        """
        l = self.block_data
        for el in l:
            el['tp'] = {}
        tot_ph_dict = {}
        for fk in self.fkeys:
            init_phase = 0
            block_ph = self.block_phases[fk]
            total_ph = np.zeros_like(block_ph)
            if fk in self.phase_offset:
                init_phase = self.phase_offset[fk]  # if expected phase at start is known, use it
                total_ph[0] = np.unwrap([init_phase, block_ph[0]])[1]
            else:
                total_ph[0] = block_ph[0]
            freq = l[0]['w'][fk]
            phase_per_block = freq * self.bl
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
        for i in range(len(self.block_data)):
            self.block_data[i]['tp']['CP'] = corr_ne_ph[i]
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
            try:
                popt, pcov = opt.curve_fit(exp_decay, t, amps, initial_guess[i])
                T2[i] = 1/popt[1]
            except RuntimeError:
                T2[i] = 'nan'
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

        t1 = self.time[self.block_mask]
        # renumber so last point is 0 to find phase at the end of the section
        t2 = self.time[self.block_mask] - self.time[-1]
        bf1 = np.array([np.ones_like(t1), t1]).transpose()
        bf2 = np.array([np.ones_like(t2), t2]).transpose()
        for f in self.fkeys:
            phases = np.array(self.total_phases[f])[self.block_mask]
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
        ax.plot(self.time, [i['amp'][fk] for i in self.block_data])
        ax.set_ylabel('{} Amplitude [V]'.format(fk + 'e'))
        ax.set_xlabel('Time [s]')
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.17)
        ax.set_title('Run {}, block length {}'.format(self.file_name, self.bl))
        return fig, ax

    def plot_phase(self, fk):
        time = self.time[self.block_mask]
        fig, ax = plt.subplots()
        real_phase = np.array([i['tp'][fk] for i in self.block_data])
        fitting_phase = self.phase_start[fk] + self.freq[fk] * time
        ax.plot(time, real_phase[self.block_mask], 'o', time, fitting_phase, '-')
        ax.legend(['phase', 'fitting'])
        ax.set_ylabel('{} Phase'.format(fk))
        ax.set_xlabel('Time [s]')
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.17)
        ax.set_title(str(self.run_number).zfill(4) + '-' + str(self.shot_number).zfill(3) + '-' + self.label)
        return fig, ax

    def plot_phase_res(self):
        """ Plot Ne, He, corrected residuals after linear fit """

        r = self.residuals
        t = self.time[self.block_mask]
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(t, r['N'], '-o')
        ax[0].set_ylabel('Ne')
        # ax[0].set_ylabel('Ne phase \nres. [rad]')
        ax[1].plot(t, r['H'], '-o')
        ax[1].set_ylabel('He')
        # ax[1].set_ylabel('He phase \nres. [rad]')
        ax[2].plot(t, r['CP'], '-o')
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
        rs = self.residuals[f]
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
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)
        file_path = os.path.join(self.outdir, file_name + ext.sc_out)

        f = open(file_path, 'w')
        output_dict = {'fkeys': self.fkeys, 'freq': self.freq,
                       'freq_err': self.freq_err, 'residuals': self.residuals,
                       'T2': self.T2, 'amps': self.amps, 'phase_start': self.phase_start,
                       'phase_end': self.phase_end, 'phase_err': self.phase_err,
                       'block length': self.bl, 'n_blocks': self.n_blocks,
                       'phase_end_time': self.phase_end_time}
        json_output = jsont.dumps(output_dict)
        f.write(json_output)
        f.close()
        return


class ScoReader:
    """ Class for reading .sco files output by SClist """

    def __init__(self, run_number, shot_number, label):
        self.label = label
        self.run_number = '{:04d}'.format(int(run_number))
        self.shot_number = '{:03d}'.format(int(shot_number))
        self.file_name = '_'.join([self.run_number, self.shot_number])+label+ext.sc_out
        self.full_name = os.path.join(lp.scodir, self.run_number, self.file_name)
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
        self.n_blocks = len(self.residuals[self.fkeys[0]])
        self.phase_end_time = self.bl * (self.n_blocks - 1)  # time of phase_end point.
