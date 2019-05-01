import numpy as np
from string import ascii_lowercase
from matplotlib import pyplot as plt
import os
import json_tricks as jsont

from gyro_analysis.rawdata import RawData
from gyro_analysis.rawfitter import RawFitter
from gyro_analysis.scfitter import SClist
from gyro_analysis.scfitter import ScoReader
from gyro_analysis.shotinfo import ShotInfo
from gyro_analysis import ddict, default_freq
import gyro_analysis.local_path as lp

# todo:  Check detection times (Plot vs Laser file?); load detection times from config file
# todo:  Check phase residuals

HENE_RATIO = 9.650

class ShotProcess:
    """ Analyze a shot with dark time

    labelling conventions:
    detection times ldet= 'a'...'z'
    dark times = 'ab' for dark time between 'a' and 'b'
    """

    def __init__(self, run_number, shot_number, detection_times=None, block_outlier={}, fitting_paras=ddict,
                 check_raw=False, check_phases=False):
        """ detection_times is 2xN list of detection times

        keywords 'check_xxx' allow user to have validation plots displayed
        """
        self.det_times = detection_times
        self.parameters = fitting_paras
        self.run_number = '{:04d}'.format(int(run_number))
        self.shot_number = '{:03d}'.format(int(shot_number))
        self.shotdir_run = os.path.join(lp.shotdir, self.run_number)
        self.n_det = len(self.det_times)
        self.narr = np.arange(self.n_det)
        self.det_labels = ascii_lowercase[:self.n_det]
        self.ok = ['y', 'Y']
        self.file_name = '_'.join([self.run_number, self.shot_number])
        self.block_outlier = dict.fromkeys(self.det_labels, [])
        self.block_outlier.update(block_outlier)
        self.abs_res_max = np.zeros(self.n_det)
        self.dark_time_dict = self.init_dark_time()
        self.process_rdt_to_rfo()
        if check_raw:
            self.check_raw_res()
        tdict = self.process_rfo_to_sco()
        self.dark_time_dict.update(tdict)
        if check_phases:
            self.check_phase_res()
        self.sco_files = {l: ScoReader(self.run_number, self.shot_number, l) for l in self.det_labels}
        self.detection_time_dict = self.load_detection_time()
        self.shotinfo = ShotInfo(self.run_number, self.shot_number)
        self.output_dict = {**self.detection_time_dict, **self.dark_time_dict, 'shotinfo': self.shotinfo.__dict__}
        self.process_sco_to_shd()

    def check_init(self):
        """ Validate parameters """
        print('check_init not yet implemented')
        return

    def check_raw_res(self):
        """ Display max residuals from each raw data fit """
        self.plot_abs_res_max()
        res_ok = input("Are all detection times okay, Y or N? [Y] ") or 'Y'
        if not res_ok in self.ok:
            raise ValueError('Data glitch in {}; code not yet able to handle glitches.')
        return

    def check_phase_res(self):
        """ Show residuals from phase fits """
        print('check_phase_res not yet implemented')
        return

    def process_rdt_to_rfo(self):
        """ run fitters on each detection time and write output files """
        fp = self.parameters
        for i in self.narr:
            fp['roi'] = self.det_times[i]
            rd = RawData(self.run_number, self.shot_number, fp)
            rf = RawFitter(rd)
            rf.process_blocks()
            self.abs_res_max[i] = rf.abs_res_max
            rf.write_json(l=self.det_labels[i])
        return

    def process_rfo_to_sco(self):
        """ fit sco files to find frequencies and phases during each detection period """
        temp_dark = self.dark_time_dict
        l = self.det_labels[0]
        sc = [SClist(self.run_number, self.shot_number, l, self.block_outlier[l])]
        sc[0].write_json()
        for i in self.narr[1:]:
            lprev = self.det_labels[i-1]
            lcurr = self.det_labels[i]
            scprev = sc[i-1]
            drop = self.block_outlier[lcurr]
            ldark = lprev+lcurr
            len_prev = self.det_times[i-1][1] - self.det_times[i-1][0]
            time_slip = len_prev - scprev.phase_end_time  # missing bit of time between phase_end and start of dark time
            time_dark = self.dark_time_dict[ldark]['time_diff']
            time_offset = time_dark + time_slip
            phase_guess = {sp: scprev.phase_end[sp] + time_offset*scprev.freq[sp] for sp in scprev.fkeys}
            phase_guess.pop('CP', None)
            sc.append(SClist(self.run_number, self.shot_number, lcurr, phase_offset=phase_guess, drop_blocks=drop))
            sc = sc[i]
            sc.write_json()
            # todo: correct phase error; record absolute phases
            for sp in scprev.fkeys:
                phase_end_prev = scprev.phase_end[sp] + time_slip*scprev.freq[sp]
                phdiff = sc.phase_start[sp] - phase_end_prev
                phdiff_err = np.sqrt(scprev.phase_err[sp] ** 2 + sc.phase_err[sp] ** 2)
                temp_dark[ldark]['phase_diff'][sp] = phdiff
                temp_dark[ldark]['phase_diff_err'][sp] = phdiff_err
                temp_dark[ldark]['freq'][sp] = phdiff/time_dark
                temp_dark[ldark]['freq_err'][sp] = phdiff_err/time_dark
        return temp_dark

    def process_sco_to_shd(self):
        """ determine dark time frequencies; load and record relevant light time values """
        self.write_json()
        return

    def init_dark_time(self):
        """ compute dark time parameters and initialize the dark time dict """
        temp_dict = {}
        for i in self.narr[:-1]:
            k1 = self.det_labels[i]
            k2 = self.det_labels[i + 1]
            key = k1 + k2
            dark_time_length = self.det_times[i + 1][0] - self.det_times[i][1]
            temp_dict[key] = dict(time_diff=dark_time_length, time_start=self.det_times[i][1],
                                       time_end=self.det_times[i + 1][0],
                                       phase_diff={}, phase_diff_err={},
                                       freq={}, freq_err={})
            return temp_dict

    def load_detection_time(self):
        """ loading detection times """
        detection_time_dict = {}
        scos = self.sco_files
        det = self.det_times
        for i in self.narr:
            l = self.det_labels[i]
            detection_time_dict[l] = dict(time_start=det[i][0], time_end=det[i][1])
            detection_time_dict[l].update(scos[l].__dict__)
            detection_time_dict[l]['abs_res_max'] = self.abs_res_max[i]
        return detection_time_dict

    def write_json(self, l=''):
        """ write json object to file """
        file_name = self.file_name + l
        if not os.path.isdir(self.shotdir_run):
            os.makedirs(self.shotdir_run)
        file_path = os.path.join(self.shotdir_run, file_name + lp.sp_ex_out)
        od = self.output_dict
        f = open(file_path, 'w')
        json_output = jsont.dumps(od)
        f.write(json_output)
        f.close()

    def plot_abs_res_max(self):
        fig, ax = plt.subplots()
        ax.plot(self.abs_res_max)
        ax.set_xlabel('Detection Time')
        ax.set_ylabel('Residual')
        ax.set_title('Maximum Residual Seen in Block Fitting')
        plt.show()

so = ShotProcess(33, 0, [[25,125], [225, 325]])
