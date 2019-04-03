import numpy as np
from numpy import sin
from numpy import cos
from numpy import pi
from scipy import signal
from scipy import fftpack
from scipy import optimize as opt
from matplotlib import pyplot as plt
import matplotlib as mpl
import os
import json

# todo:  FitData needs: start, end, dt, absolute start;  JSON dump.
# todo:  Class operating on list of FitData JSON load
from copy import deepcopy

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.unicode_minus'] = False
mpl.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
mpl.rcParams['axes.formatter.limits'] = (-2, 2)
# mpl.rc('text', usetex=True)

HENE_RATIO = 9.650

ddict = dict(dt=1e-3, mag=[4, 500], spins=[12, 500], roi=[20, 500], nave=1)
tdict = dict(dt=1e-3, mag=[0, 57], spins=[0, 57], roi=[0, 57], nave=1)

freqlist = ['X', '0h3n', '0h2n', '1h-2n', '1h2n', '1h-1n', '1h1n', '5n', '4n', '1.5n']
default_freq = dict(wH=15.0148 * 2 * pi, wN=1.55606 * 2 * pi)
fN = default_freq['wN'] / 2 / pi
fH = default_freq['wH'] / 2 / pi
df = default_freq

ft = 'data_18-12-14_2132_005'


class SClist:
    """
    Stores and manipulates a list of SCdat objects plus a header.

    SClist.l contains the list of SCdat objects one for each block
    SClist.fkeys  contains a list of the species whose signals are being computed.
    SClist.amps, contains amplitudes for each signal
    SClist.total_phases contains phases for each signal
    SClist.phase(time) gives inferred phase at that time
    SClist.fp (or freqs_and_phases) contains dict with keys:
        'phase_start', 'phase_start_err', 'phase_end', 'phase_end_err'
        'freq_start', 'freq_start_err', 'freq_end', 'freq_end_err'
        Each value is a dict with frequencies.
    SClist.res contains residuals 'res_start', 'res_end' for fits done with t_start = 0 or t_end = 0.


    """

    def __init__(self, scdir, name):
        self.hene_ratio = HENE_RATIO
        self.name = name
        self.ext = '.scf'
        self.path = scdir
        self.fullname = os.path.join(self.path, self.name + self.ext)
        with open(self.fullname, 'r') as read_data:
            print('loading file:' + self.fullname)
            data = json.load(read_data)
        self.hdr = data[-1]
        self.l = data[:-1]
        self.keys = self.l[0].keys()
        self.fkeys = list(self.l[0]['w'].keys())
        self.dt = self.l[0]['dt']
        self.amp_phase()
        self.time = np.array([i['start'] * self.dt for i in self.l])
        self.amps = self.collect_amps()
        self.block_phases = self.collect_block_phases()
        self.bl = (self.l[0]['end'] - self.l[0]['start']) * self.dt
        self.total_phases = self.collect_total_phases()
        self.ne_corr = self.corrected_phases()
        self.fkeys.append('CP')
        freqs_and_phases, residuals = self.fit_phases()
        self.freqs_and_phases = freqs_and_phases
        self.fp = freqs_and_phases
        self.res_dict = residuals
        self.freqs = self.fp['freq_start']
        self.freq_errs = self.fp['freq_start_err']
        self.phase_start = self.fp['phase_start']
        self.phase_end = self.fp['phase_end']
        self.phase_errs = self.fp['phase_err']
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

    def fit_phases(self):
        """
        Use linear regression to fit phases to a line (frequency), start phase and end phase.

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

    def phase(self, time):
        ph_out = {}
        for f in self.fkeys:
            ph_out[f] = self.freqs[f]*time + self.init_phases[f]
        return ph_out

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
        ax.set_title('Run {}, block length {}'.format(self.name, self.bl))
        return fig, ax

    def plot_phase(self, fk):
        fig, ax = plt.subplots()
        ax.plot(self.time, [i['tp'][fk] for i in self.l])
        ax.set_ylabel('{} Amplitude [V]'.format(fk + 'e'))
        ax.set_xlabel('Time [s]')
        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(bottom=0.17)
        ax.set_title('Run {}, block length {}'.format(self.name, self.bl))
        return fig, ax

    def plot_phase_res(self):
        """ Plot Ne, He, corrected residuals after linear fit """

        r = self.res_phases
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
