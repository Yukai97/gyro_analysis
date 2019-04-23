import os

homedir = r"C:\Users\Romalis Group\Desktop\New Data"
rawdir = os.path.join(homedir, 'Raw data')
infodir = os.path.join(homedir, 'Info data')
scdir = os.path.join(homedir, 'SCO data')
shotdir = os.path.join(homedir, 'Shot data')

rd_ex_in = '.rdt'
rf_ex_in = '.rdt'
rf_ex_out = '.rfo'
sc_ex_in = '.rfo'
sc_ex_out = '.sco'
dt_ex_in = '.sco'
dt_ex_out = '.shd'
shot_ex_header = '.hdr'