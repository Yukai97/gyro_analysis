import os
import re
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from gyro_analysis import ddict, ave_array

#todo: PEP 8 for package
#todo: move rc params settings

class RawData:
    """
    Loads voltage data from the specified file. ddict contains parameters

    example: ddict = dict(dt=1e-3, mag=[4, 500], spins=[12, 500], roi=[20, 500], nave=1)
    dt: time step between volatge records
    mag: start and end time of magnetometer  # currently not used
    spins: start and end time of spin precession  # currently not used
    roi: region of interest to analyze
    """
    # todo: get_data load: try except loadtxt; user input if fname absent

    def __init__(self, homedir, rawdir, fdir, scdir, filename=None, hfile=ddict):
        self.homedir = homedir
        self.rawdir = rawdir
        self.scdir = scdir
        self.fdir = fdir
        self.dir = os.path.join(self.rawdir, fdir)
        self.name = filename
        self.header = hfile  # eventually, make header files and use self.get_hdr(hfile)
        self.dt = self.header['dt']
        self.fs = 1 / self.dt
        self.all_data = self.get_data()
        self.data = self.roi_data()
        self.nave = self.header['nave']
        self.time = self.make_time()
        self.time_stamp = self.get_time_stamp()
        if self.nave > 1:
            self.ave_data()  # changes self.data, self.dt self.fs and self.time
        # self.tlength = self
        self.psd = None

    # def get_hdr(self, hfile):
    #     default_header = dict(dt=1e-3, mag=[4, 500], spins=[12, 500], roi=[20, 500])
    #     if hfile is None:
    #         return default_header
    #     if type(hfile) is dict:
    #         return hfile
    #     else:
    #         return hfile
    #         print('Need to learn to read header files')
    #         # todo: design header file format syntax and load here; ensure there is a dt term and roi term

    def get_time_stamp(self):
        if 'local_time' in self.header:
            loc = self.header['local_time']
        else:
            full_file = os.path.join(self.dir, self.name)
            t = os.path.getctime(full_file)
            tmptime = time.localtime(t)
            return time.strftime('%Y-%m-%d', tmptime)
        return loc

    # set up class instance

    def get_data(self):
        while True:
            full_file = os.path.join(self.dir, self.name)
            try:
                raw_data = np.loadtxt(full_file)
                break
            except FileNotFoundError:
                inp = [0,0]
                print('File name {} in directory {} not found.'.format(self.name, self.dir))
                inp[0] = input('Enter run directory [{}] or x to exit:'.format(self.fdir)) or self.fdir
                inp[1] = input('Enter file name  to analyze [{}] or x to exit): '.format(self.name)) or self.name
                if 'x' in inp or 'X' in inp:
                    raise FileNotFoundError('Failed to open raw data file')
                self.fdir = inp[0]
                self.name = inp[1]
                self.dir = os.path.join(self.rawdir, self.fdir)
        if raw_data.ndim == 1:
            return raw_data
        else:
            return raw_data[:, -1]

    def roi_data(self):
        roi = self.header['roi']
        fs = self.fs
        return self.all_data[int(roi[0] * fs):int(roi[1] * fs)]

    def ave_data(self):
        nave = self.nave
        self.data = ave_array(self.data, nave)
        self.dt = self.dt * nave
        self.fs = 1 / self.dt

    def make_time(self):
        roi = self.header['roi']
        return np.arange(roi[0] * self.fs, roi[1] * self.fs) * self.dt

    def plot_all_data(self):
        plt.plot(np.arange(0, len(self.all_data)) * self.dt, self.all_data)
        plt.xlabel('time')
        plt.ylabel('voltage')

    def plot_roi_data(self):
        plt.plot(self.time, self.data)
        plt.xlabel('time')
        plt.ylabel('voltage')

    # look at data

    def make_psd(self, nseg=None):
        ntot = len(self.data)
        fs = 1 / self.dt
        if not nseg:
            nseg = 8
        npts = 2 ** (np.floor(np.log2(ntot / nseg)))
        self.psd = signal.welch(self.data, fs, nperseg=npts)

    def plot_psd(self, fig=None, ax=None):
        if not self.psd:
            self.make_psd(8)

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.psd[0], np.sqrt(self.psd[1]))
        ax.set_xlabel('Hz')
        ax.set_ylabel('$\sqrt{V^2/Hz}$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([0.1, 30])
        return fig, ax


