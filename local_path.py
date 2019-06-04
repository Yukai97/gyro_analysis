import os



class paths:

    homedir = r"D:\gyro_data\Data"
    rawdir = os.path.join(homedir, 'Raw data')
    infodir = os.path.join(homedir, 'Info data')
    rfodir = os.path.join(homedir, 'RFO data')
    scodir = os.path.join(homedir, 'SCO data')
    shotdir = os.path.join(homedir, 'Shot data')


class extensions:
    rd_in = '.rdt'
    rf_in = '.rdt'
    rf_out = '.rfo'
    sc_in = '.rfo'
    sc_out = '.sco'
    sp_in = '.sco'
    sp_out = '.shd'
    shot_header = '.hdr'

ext = extensions
lp = paths

homedir = lp.homedir
rawdir = lp.rawdir
infodir = lp.infodir
rfodir = lp.rfodir
scodir = lp.scodir
shotdir = lp.shotdir

rd_ex_in = ext.rd_in
rf_ex_in = ext.rf_in
rf_ex_out = ext.rf_out
sc_ex_in = ext.sc_in
sc_ex_out = ext.sc_out
sp_ex_in = ext.sp_in
sp_ex_out = ext.sp_out
shot_ex_data = ext.sp_out
shot_ex_header = ext.shot_header

rd_in = ext.rd_in
rf_in = ext.rf_in
rf_out = ext.rf_out
sc_in = ext.sc_in
sc_out = ext.sc_out
sp_in = ext.sp_in
sp_out = ext.sp_out
shot_header = ext.shot_header