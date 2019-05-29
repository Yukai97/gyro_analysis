import os




homedir = r"C:\Users\Romalis Group\Desktop\New Data"
rawdir = os.path.join(homedir, 'Raw data')
infodir = os.path.join(homedir, 'Info data')
rfodir = os.path.join(homedir, 'RFO data')
scodir = os.path.join(homedir, 'SCO data')
shotdir = os.path.join(homedir, 'Shot data')

rd_in = '.rdt'
rf_in = '.rdt'
rf_out = '.rfo'
sc_in = '.rfo'
sc_out = '.sco'
sp_in = '.sco'
sp_out = '.shd'
shot_header = '.hdr'

class paths:
    #homedir = r"C:\Users\Romalis Group\Desktop\New Data"
    homedir = r"/Users/William/Library/Mobile Documents/com~apple~CloudDocs/Work/" \
              r"Active Projects/HE-NE/Code/B30/Data"
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

