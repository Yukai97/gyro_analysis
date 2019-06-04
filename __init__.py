""" Written by William Terrano and Yukai Lu March 2019 """

import matplotlib as mpl
import numpy as np
import os
from numpy import pi

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.unicode_minus'] = False
mpl.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
# mpl.rc('text', usetex=True)

ddict = dict(dt=1e-3, mag=[4, 500], spins=[12, 500], roi=[25, 500], nave=1)
tdict = dict(dt=1e-3, mag=[0, 57], spins=[0, 57], roi=[0, 57], nave=1)

freqlist = ['X', '0h3n', '1h-2n', '1h2n', '1h-1n', '1h1n', '5n', '4n', '1.5n']
default_freq = dict(wH=15.0148 * 2 * pi, wN=1.55606 * 2 * pi)
fN = default_freq['wN'] / 2 / pi
fH = default_freq['wH'] / 2 / pi
df = default_freq
kappa_ne_rb = 38
kappa_he_rb = 5
gamma_he = 2*np.pi*32.43409966
gamma_ne = 2*np.pi*32.43409966/9.649*10**6
calibration_factor = 22792.573 * 10**4
mu_0 = 4*pi*10**(-7)


def ave_array(arr_to_ave, npts):
    arr = np.array(arr_to_ave)
    la = len(arr)
    ax = la % npts
    ashape = arr[:la-1*ax].reshape(-1, npts)
    return np.mean(ashape, 1)


__all__ = ['rawdata', 'rawfitter', 'scfitter', 'analyzer', 'shotinfo', 'local_path', 'ddict', 'tdict', 'freqlist',
           'default_freq', 'fN', 'fH', 'df', 'ave_array', 'kappa_he_rb', 'kappa_ne_rb', 'gamma_he', 'gamma_ne',
           'calibration_factor', 'mu_0']
