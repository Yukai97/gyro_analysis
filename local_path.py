import os

homedir = r"C:\Users\Romalis Group\Desktop\New Data"
rawdir = os.path.join(homedir, 'Raw data')
infodir = os.path.join(homedir, 'Info data')
scdir = os.path.join(homedir, 'SC data')
wddir = os.path.join(homedir, 'WD data')
shotdir = os.path.join(homedir, 'Shot data')

rd_ex_in = '.rdt'
rf_ex_in = '.rdt'
rf_ex_out = '.scf'
sc_ex_in = '.scf'
sc_ex_out = '.wdf'
dt_ex_in = '.wdf'
dt_ex_out = '.shd'
shot_ex_header = 'hdr'
