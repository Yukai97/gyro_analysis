import os
import re
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from gyro_analysis import ddict, ave_array
from gyro_analysis import shotinfo
from gyro_analysis.local_path import *


class RawData:
    """Loads voltage data from the specified file.

    ddict contains parameters
    example: ddict = dict(dt=1e-3, mag=[4, 500], spins=[12, 500], roi=[20, 500], nave=1)
    dt: time step between volatge records
    mag: start and end time of magnetometer  # currently not used
    spins: start and end time of spin precession  # currently not used
    roi: region of interest to analyze
    """
    # todo: get_data load: try except loadtxt; user input if fname absent


    def __init__(self, run_number, filename=None, fitting_paras=ddict):
        self.shotinfo = shotinfo.ShotInfo(run_number, filename)
        self.path = {'homedir': homedir, 'rawdir': rawdir, 'infordir': infodir, 'scdir': scdir, 'shotdir': shotdir}
        self.name = filename
        self.ext = '.rdt'
        self.run_number = run_number
        self.rawdir_run = os.path.join(self.path['rawdir'], run_number)
        self.fitting_paras = fitting_paras  # eventually, make header files and use self.get_hdr(hfile)
        self.dt = self.fitting_paras['dt']
        self.fs = 1 / self.dt
        self.all_data = self.load_data()
        self.data = self.roi_data()
        self.nave = self.fitting_paras['nave']
        self.time = self.make_time()
        if self.nave > 1:
            self.ave_data()  # changes self.data, self.dt self.fs and self.time
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

    # set up class instance

    def load_data(self):
        while True:
            full_file = os.path.join(self.rawdir_run, self.name + self.ext)
            try:
                raw_data = np.loadtxt(full_file)
                break
            except FileNotFoundError:
                inp = [0, 0]
                file_name = self.name + self.ext
                print('File name {} in directory {} not found.'.format(file_name, self.rawdir_run))
                inp[0] = input('Enter run directory [{}] or x to exit:'.format(self.run_number)) or self.run_number
                inp[1] = input('Enter file name  to analyze [{}] or x to exit): '.format(file_name)) or file_name
                if 'x' in inp or 'X' in inp:
                    raise FileNotFoundError('Failed to open raw data file')
                self.run_number = inp[0]
                self.name = inp[1].split('.')[0]
                self.ext = inp[1].split('.')[1]
                self.rawdir_run = os.path.join(self.path['rawdir'], self.run_number)
        if raw_data.ndim == 1:
            return raw_data
        else:
            return raw_data[:, -1]

    def roi_data(self):
        roi = self.fitting_paras['roi']
        fs = self.fs
        return self.all_data[int(roi[0] * fs):int(roi[1] * fs)]

    def ave_data(self):
        nave = self.nave
        self.data = ave_array(self.data, nave)
        self.dt = self.dt * nave
        self.fs = 1 / self.dt

    def make_time(self):
        roi = self.fitting_paras['roi']
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

    def plot_psd(self, xlim=[0.1, 30], fig=None, ax=None):
        if not self.psd:
            self.make_psd(8)

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(self.psd[0], np.sqrt(self.psd[1]))
        ax.set_xlabel('Hz')
        ax.set_ylabel('$\sqrt{V^2/Hz}$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(xlim)
        return fig, ax


