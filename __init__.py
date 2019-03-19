import matplotlib as mpl
import numpy as np
import os
from numpy import pi

# todo: convert all visualization methods to plt_xxx
# todo: documentation, info and dictionary keys + explanation & help methods
# todo: make jupyter notebooks testing block-length dependence of fit
# todo: look at fit to corrected phases

mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.unicode_minus'] = False
# mpl.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
# mpl.rc('text', usetex=True)

HOMEDIR = r"C:\Users\Romalis Group\Desktop\New Data"
RAWDIR = os.path.join(HOMEDIR, 'Raw data')
RUNDIR = '0012'

ddict = dict(dt=1e-3, mag=[4, 500], spins=[12, 500], roi=[25, 500], nave=1)
tdict = dict(dt=1e-3, mag=[0, 57], spins=[0, 57], roi=[0, 57], nave=1)

freqlist = ['X', '0h3n', '0h2n', '1h-2n', '1h2n', '1h-1n', '1h1n', '5n', '4n', '1.5n']
default_freq = dict(wH=15.0148 * 2 * pi, wN=1.55606 * 2 * pi)
fN = default_freq['wN'] / 2 / pi
fH = default_freq['wH'] / 2 / pi
df = default_freq


def ave_array(arr_to_ave, npts):
    arr = np.array(arr_to_ave)
    la = len(arr)
    ax = la % npts
    ashape = arr[:la-1*ax].reshape(-1, npts)
    return np.mean(ashape, 1)

# class block_settings(block_length, nyquist_freq)

# def fit_blocks(fname, dt=1e-3):
#    data = np.loadtxt(currdir+fname)

# if __name__ == "__main__":
#     ff = 'data_18-12-14_2132_005.txt'
#     r = RawData(filename=ff, hfile=ddict)
#     b = SCfitter(r)
#     b.fit_blocks()


# fl = ['data_18-12-14_2132_005.txt',
#       'data_18-12-14_2242_009.txt',
#       'data_18-12-14_2318_011.txt',
#       'data_18-12-15_0011_014.txt',
#       'data_18-12-15_0104_017.txt',
#       'data_18-12-15_0140_019.txt',
#       'data_18-12-15_0157_020.txt',
#       'data_18-12-15_0326_025.txt',
#       'data_18-12-15_0716_038.txt',
#       'data_18-12-15_1013_048.txt']
