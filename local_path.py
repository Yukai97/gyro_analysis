import os

homedir = r"D:\gyro_data\Data"
rawdir = os.path.join(homedir, 'Raw data')
infodir = os.path.join(homedir, 'Info data')
rfodir = os.path.join(homedir, 'RFO data')
scodir = os.path.join(homedir, 'SCO data')
shotdir = os.path.join(homedir, 'Shot data')

rd_ex_in = '.rdt'
rf_ex_in = '.rdt'
rf_ex_out = '.rfo'
sc_ex_in = '.rfo'
sc_ex_out = '.sco'
sp_ex_in = '.sco'
sp_ex_out = '.shd'
shot_ex_data = '.shd'
shot_ex_header = '.hdr'
