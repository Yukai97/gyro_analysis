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

homedir = r"C:\Users\Romalis Group\Desktop\New Data"
rawdir = os.path.join(homedir, 'Raw data')
scdir = os.path.join(homedir, 'SC data')
rundir = '0012'

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
    SClist.


    """

    def __init__(self, name):
        self.hene_ratio = HENE_RATIO
        self.name = name
        self.ext = '.scf'
        self.path = scdir
        self.fullname = os.path.join(self.path, self.name + self.ext)
        with open(self.fullname, 'r') as read_data:
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
        self.freqs, self.init_phases, self.phase_res = self.fit_phases()

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
                el['bp'][f] = (np.arctan2(a, b) + 2 * pi) % (2 * pi)
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


        :return:
        """
        fit_freqs = {}
        fit_ph = {}
        res = {}
        t = self.time
        bf = np.array([np.ones_like(t), t]).transpose()
        for f in self.fkeys:
            phases = self.total_phases[f]
            lf = opt.lsq_linear(bf, phases, tol=1e-16)
            if not lf['success']:
                raise RuntimeError('Fit of phases to line did not converge')
            freq = lf['x'][1]
            initial_phase = lf['x'][0]
            r = phases - initial_phase - freq * t
            fit_freqs[f] = freq
            fit_ph[f] = initial_phase
            res[f] = r
        return fit_freqs, fit_ph, res

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






#st = SClist(ft)
#s2 = SClist(ft + 'c')
sne = SClist(ft + 'ne')


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
