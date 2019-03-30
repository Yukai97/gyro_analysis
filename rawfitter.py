import json
import os
from copy import deepcopy
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from numpy.core.umath import sin, cos
from scipy import optimize as opt, signal

from gyro_analysis import rawdata, fN, default_freq, ave_array


class RawFitter:
    """ Initialized with RawData object, length of blocks and frequencies to fit
    (He and Ne always included)

    Fitting procedure:
    1) Divide data into blocks of length block_length
    2) Fit each segment to He & Ne amplitudes given a default He/Ne freqs
    3) Use amps from (2) to find best individual frequency for each block
    4) Use frequencies from (3) in a linear regression to find all amplitudes of interest
     --> Xe and side-band frequencies determined from fitted (3) He/Ne frequencies

    RawFitter.fit_blocks() runs the block fit on the RawData
    RawFitter.blist contains list of the RawBlock objects resulting from the fit, one for each block
    RawFitter.write_json() writes a json of the RawBlock objects to the SCfit directory.

    """

    def __init__(self, rawdata: rawdata, block_length=6 / fN,
                 freqs2fit=['X', '0h3n']):
        """

        :type rawdata: RawData class
        """
        self.raw = deepcopy(rawdata)
        self.time = self.raw.time
        self.dt = self.raw.dt
        self.bl = block_length  # in seconds
        self.freqs2fit = freqs2fit
        self.nb = int((rawdata.time[-1] - rawdata.time[0]) / block_length)
        self.base_fit = self.set_fitting_freqs(freqs2fit)
        self.blist = self.init_blocks()
        self.fit_span = self.blist[-1].end - self.blist[0].start
        self.time = self.time[:int(self.fit_span)]
        self.data, self.offset = self.remove_offset()
        self.res = np.zeros(int(self.fit_span))
        self.r_squared = None  # defined in fit_blocks
        self.res_int_list = None  # defined in fit_blocks
        self.res_int = None  # defined in fit_blocks
        # self.ind = [int(i * block_length * self.raw.fs) for i in range(self.nb)]
        self.p = np.arange(self.nb)

    def remove_offset(self):
        """ subtract quadratic offset from raw data """
        t = self.time
        d = self.raw.data[:int(self.fit_span)]
        t2 = t**2
        bf = np.array([np.ones_like(t), t, t2]).transpose()
        lf = opt.lsq_linear(bf, d, tol=1e-16)
        c0, c1, c2 = lf['x'][0], lf['x'][1], lf['x'][2]
        return d - (c0 + c1*t + c2*(t2)), [c0, c1, c2]

    def init_blocks(self):
        dt = self.dt
        blen = int(np.round(self.bl/dt))
        bl_list = []
        for i in range(self.nb):
            b = deepcopy(self.base_fit)
            start = int(blen * i)
            end = int(blen * (i + 1))
            b.start = start
            b.end = end
            b.dt = dt
            bl_list.append(b)
        return bl_list

    def set_fitting_freqs(self, f2f=None):
        if f2f is None:
            f2f = self.get_freqs()
        wantX = False
        if 'X' in f2f:
            wantX = True
        harms = list(set(f2f) - set(['X', 'C']))
        return RawBlock(wH=default_freq['wH'], wN=default_freq['wN'], wX=wantX, wHarm=harms)

    @property
    def starts(self):
        " list of start times in blocklist "
        start_mat = np.array([b.start for b in self.blist])
        return start_mat.reshape(-1)

    def block_time(self, s, e):
        return self.raw.time[s:e] - self.raw.time[s]

    def block_data(self, s, e):
        rd = self.data[s:e]
        return rd

    def get_r_squared(self, res, data):
        ss_res = np.sum(res ** 2)
        data_ave = np.mean(data)
        ss_tot = np.sum((data - data_ave) ** 2)
        r_squared = 1 - ss_res / ss_tot
        return r_squared

    def fit_blocks(self):
        r2_list = []
        res_int_list = []
        for block in self.blist:
            s, e, dt = block.start, block.end, block.dt
            t = self.block_time(s, e)  # find phase relative to block start
            data = self.block_data(s, e)  # data, offset subtracted
            block = self.fit_seg(block)
            res = data - block.eval(t)
            r_squared = self.get_r_squared(res, data)
            res_int = sum(res) * dt
            self.res[s:e] = res
            block.r_squared = r_squared
            block.res_int = res_int
            r2_list.append(r_squared)
            res_int_list.append(res_int)
        self.r_squared = np.array(r2_list)
        self.res_int_list = np.array(res_int_list)
        self.res_int = np.sum(res_int_list)
        return

    def fit_seg(self, scdat_obj):
        seg = scdat_obj
        s, e = seg.start, seg.end
        wH, wN = seg.wH, seg.wN
        a0 = self.fit_amp_seg(s, e, [wH, wN])
        w0 = self.fit_freq_seg(s, e, *a0)
        seg.update_hene(w0)
        names = seg.names
        w1 = [seg.w[i] for i in names]
        amps = self.fit_amp_seg(s, e, w1).reshape(-1, 2)
        for i in range(len(names)):
            seg.a[names[i]] = amps[i][0]
            seg.b[names[i]] = amps[i][1]
        return seg

    # fit amplitudes of arbitrary frequencies
    def fit_amp_seg(self, start, end, freqs):
        """ find sine and cosine amplitudes (nx2 array) of n freqs;  """
        t = self.block_time(start, end)
        data = self.block_data(start, end)
        bf0 = self.bf_freqs(t, freqs).transpose()
        lf = opt.lsq_linear(bf0, data, tol=1e-16)
        if not lf['success']:
            raise RuntimeError('Fit of raw data to sine & cosine')
        return lf['x']

    # noinspection PyPep8Naming
    def fit_freq_seg(self, start, end, asH, acH, asN, acN):
        """ """
        wH, wN = self.base_fit.wH, self.base_fit.wN
        block_t = self.block_time(start, end)
        data = self.block_data(start, end)
        A0 = [asH, acH, asN, acN]

        def res(w, t, d, amps):
            para_dict = dict(aH=amps[0], bH=amps[1], aN=amps[2], bN=amps[3], wH=w[0], wN=w[1])
            return self.func_hene(t, para_dict) - d

        return opt.least_squares(res, x0=[wH, wN], ftol=1e-12, args=(block_t, data, A0))['x']

    def func_hene(self, t, params):
        """ Returns a sin wt + b cos wt for a pair of frequencies  """
        aH = params['aH']
        bH = params['bH']
        wH = params['wH']
        aN = params['aN']
        bN = params['bN']
        wN = params['wN']
        return aH * sin(wH * t) + bH * cos(wH * t) + aN * sin(wN * t) + bN * cos(wN * t)

    def bf_freqs(self, t, freqs):
        """ sin/cos basis functions from arbitrary list of frequencies """
        basis = []  # basis function
        for w in freqs:
            basis.append(sin(w * t))
            basis.append(cos(w * t))
        return np.array(basis)

    def plot_fit(self, ind):
        ind = ind  # self.get_index(time)
        bl = self.blist[ind]
        start = bl.start
        end = bl.end
        t = self.block_time(start, end)
        data = self.block_data(start, end)
        fitting_data = bl.eval(t)
        fig, ax = plt.subplots()
        ax.plot(t + self.time[start], data, 'k.')
        ax.plot(t + self.time[start], fitting_data, 'b')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('signal [V]')
        fig.show()
        return fig, ax

    def plot_res(self):
        t = self.time
        res = self.res
        fig, ax = plt.subplots()
        ax.plot(t, res, 'k.')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Residuals [V]')
        fig.show()
        return fig, ax

    def plot_res_ave(self, ave_points = 30):
        """
        average residuals by ave_points before plotting, to remove high-frequency component.

        :return: figure, axes objects

        :param self:
        :param ave_points: number of points to average
        :return:
        """
        ta = ave_array(self.time, ave_points)
        resa = ave_array(self.res, ave_points)
        fig, ax = plt.subplots()
        ax.plot(ta, resa, 'k.')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Residuals [V]')
        fig.show()
        return fig, ax

    def plot_r_squared(self):
        block_number = np.arange(len(self.blist))
        r_squared = self.r_squared
        plt.plot(block_number, r_squared)
        plt.xlabel('block number')
        plt.ylabel('$R^2$')
        plt.show()

    def make_raw_psd(self, nseg=8):
        dat = self.raw.data
        ntot = len(dat)
        fs = 1 / self.dt
        npts = 2 ** (np.floor(np.log2(ntot / nseg)))
        return signal.welch(dat, fs, nperseg=npts)

    def make_res_psd(self, nseg=8):
        res = self.res
        ntot = len(res)
        fs = 1 / self.dt
        npts = 2 ** (np.floor(np.log2(ntot / nseg)))
        return signal.welch(res, fs, nperseg=int(npts))

    def plot_res_psd(self, xlim=[0.01, 30], nseg=8):
        psd = self.make_res_psd(nseg)
        fig, ax = plt.subplots()
        ax.plot(psd[0], np.sqrt(psd[1]))
        ax.set_xlabel('Hz')
        ax.set_ylabel('$\sqrt{V^2/Hz}$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(xlim)
        return fig, ax

    def write_json(self, l=''):

        "write blist (output of fit_blocks()) to a json file of same name as RawData file + l"
        fname = os.path.splitext(self.raw.name)[0]+l
        now = datetime.now()

        def default_json(o):
            return o.__dict__

        hdr = {'name': fname, 'block_length': self.bl, 'dt': self.raw.dt,
               'runtime_stamp': str(self.raw.time_stamp),
               'writetime': now.strftime('%Y-%m-%d %H:%M:%S'),
               'offset': self.offset, 'freqlist': self.freqs2fit, 'ddict': self.raw.header,
               'default_freq': default_freq}

        wn = os.path.join(self.raw.scdir, fname + '.scf')
        f = open(wn, 'w')
        lst = deepcopy(self.blist)
        lst.append(hdr)
        json.dump(lst, f, default=default_json)
        f.close()
        return

    def export_res(self, res_path, l=''):

        """
        export residuals to files. The first row is time and the second row is res. 
        """

        fname = os.path.splitext(self.raw.name)[0]+l
        full_path = os.path.join(res_path, fname + '.res')
        t = self.time
        res = self.res
        np.savetxt(full_path, (t, res))
        return


class RawBlock:
    """
    Class storing data related to sine and cosine fit to the data


    syntax:
    wH = helium frequency (rad)
    wX = xenon frequency (rad)
    wN = neon frequency (rad)
    wC = calibration tone frequency (rad) typically 8 Hz *2*pi
    w_nH_mN means n*wH + m*wN and represents that combination of harmonics.
    n will always be 0 or 1 for what we are interested in.


    """

    def __init__(self, wH=default_freq['wH'], wN=default_freq['wN'], wX=False, wC=False, wHarm=[]):
        self.hene_ratio = 9.650
        self.hexe_ratio = 2.754
        self.xene_ratio = self.hene_ratio / self.hexe_ratio
        self.fund = ['H', 'N', 'X', 'C']  # fundamental frequencies
        self.w = self.set_freqs_fund(wH, wN, wX, wC)  # set the fundamental frequencies to include in fit
        self.set_freqs_harm(wHarm)  # Add any harmonics of interest
        init_amps = self.set_amps()
        self.a = init_amps[0]
        self.b = init_amps[1]
        self.r_squared = None
        self.res_int = None
        self.start = None
        self.end = None
        self.dt = None
        self.local_time = None

    def set_freqs_fund(self, wH, wN, wX, wC):
        fdict = dict(H=wH, N=wN)
        if wX:
            fdict['X'] = self.xene_ratio * wN
        if wC:
            fdict['C'] = wC
        return fdict

    def update_hene(self, w):
        self.w['H'] = w[0]
        self.w['N'] = w[1]
        if 'X' in self.w:
            self.w['X'] = self.xene_ratio * self.wN
        harms = list(set(self.names) - set(self.fund))
        self.set_freqs_harm(harms)

    def set_freqs_harm(self, wHarm):
        harms = self.parse_harm(wHarm)
        self.w.update(harms)
        return

    def parse_harm(self, wHarm):
        import re
        w_out = {}
        wH, wN, wX, wC = self.wH, self.wN, self.wX, self.wC
        for s in wHarm:
            try:
                nH = float(re.search(r'([-+]?\d+\.?\d*)h', s, re.I).group(1))
            except AttributeError:
                nH = 0
            try:
                nN = float(re.search(r'([-+]?\d+\.?\d*)n', s, re.I).group(1))
            except AttributeError:
                nN = 0
            try:
                nX = float(re.search(r'([-+]?\d+\.?\d*)x', s, re.I).group(1))
            except AttributeError:
                nX = 0
            try:
                nC = float(re.search(r'([-+]?\d+\.?\d*)c', s, re.I).group(1))
            except:
                nC = 0
            w = nH * wH + nN * wN + nX * wX + nC * wC
            w_out[s] = w
        return w_out

    def set_amps(self):
        asin = {}
        bcos = {}
        for name in self.w:
            asin[name] = None
            bcos[name] = None
        return asin, bcos

    def eval(self, t):
        val = 0
        a, b, w = self.a, self.b, self.w
        for l in self.names:
            val += a[l] * sin(w[l] * t) + b[l] * cos(w[l] * t)
        return val

    def disp(self):
        print("Frequencies in Fit: {}".format(self.w))
        print("Sine Amplitides: {}".format(self.a))
        print("Cosine Amplitudes: {}".format(self.b))

    @property
    def wX(self):
        try:
            w = self.w['X']
        except KeyError:
            w = self.xene_ratio * self.wN
        return w

    @property
    def wC(self):
        try:
            w = self.w['C']
        except KeyError:
            w = 0
        return w

    @property
    def wH(self):
        return self.w['H']

    @property
    def wN(self):
        return self.w['N']

    @property
    def names(self):
        return list(self.w)
