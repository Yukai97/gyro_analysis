import numpy as np
from string import ascii_lowercase
from matplotlib import pyplot as plt
import os
import json_tricks as jsont

from gyro_analysis.rawdata import RawData
from gyro_analysis.rawfitter import RawFitter
from gyro_analysis.scfitter import SClist
from gyro_analysis.scfitter import WdfReader
from gyro_analysis import ddict
import gyro_analysis.local_path as lp

# todo:  Check detection times (Plot vs Laser file?); load detection times from config file
# todo:  Check phase residuals


class DarkTime:
    """ Analyze a shot with dark time

    labelling conventions:
    detection times ldet= 'a'...'z'
    dark times = 'ab' for dark time between 'a' and 'b'
    """

    def __init__(self, run_number, shot_number, detection_times=None, fitting_paras=ddict,
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
        self.labels = ascii_lowercase[:self.n_det]
        self.ok = ['y', 'Y']
        self.file_name = '_'.join([self.run_number, self.shot_number])
        self.check_init()
        self.abs_res_max = np.zeros(self.n_det)
        self.process_rdt_to_scf()
        if check_raw:
            self.check_raw_res()
        self.process_scf_to_wdf()
        if check_phases:
            self.check_phase_res()
        self.wdfs = {l: WdfReader(self.run_number, self.shot_number, l) for l in self.labels}
        self.dark_time_dict = self.calc_dark_time()
        self.detection_time_dict = self.load_detection_time()
        self.output_dict = {**self.detection_time_dict, **self.dark_time_dict}
        self.process_wdf_to_shd()

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

    def process_rdt_to_scf(self):
        """ run fitters on each detection time and write output files """
        fp = self.parameters
        for i in self.narr:
            fp['roi'] = self.det_times[i]
            rd = RawData(self.run_number, self.shot_number, fp)
            rf = RawFitter(rd)
            self.abs_res_max[i] = rf.abs_res_max
            rf.process_blocks()
            rf.write_json(l=self.labels[i])
        return

    def process_scf_to_wdf(self):
        """ fit sco files to find frequencies and phases during each detection period """
        for l in self.labels:
            sc = SClist(self.run_number, self.shot_number, l)
            sc.write_json()
        return

    def process_wdf_to_shd(self):
        """ determine dark time frequencies; load and record relevant light time values """
        self.write_json()
        return

    def calc_dark_time(self):
        """ determine dark time frequencies """
        dark_time_dict = {}
        wdfs = self.wdfs
        for i in self.narr[:-1]:
            k1 = self.labels[i]
            k2 = self.labels[i + 1]
            key = k1 + k2
            time_diff = self.det_times[i + 1][0] - self.det_times[i][1]
            dark_time_dict[key] = dict(time_diff=time_diff, time_start=self.det_times[i][1],
                                       time_end=self.det_times[i + 1][0],
                                       phase_diff={}, phase_diff_err={},
                                       freq={}, freq_err={})
            for sp in wdfs[k1].fkeys:
                phdf = wdfs[k2].phase_start[sp] - wdfs[k1].phase_end[sp]
                phdf_err = np.sqrt(wdfs[k2].phase_err[sp] ** 2 + wdfs[k1].phase_err[sp] ** 2)
                dark_time_dict[key]['phase_diff'][sp] = phdf
                dark_time_dict[key]['phase_diff_err'][sp] = phdf_err
                dark_time_dict[key]['freq'][sp] = phdf/time_diff
                dark_time_dict[key]['freq_err'][sp] = phdf_err/time_diff
        return dark_time_dict

    def load_detection_time(self):
        """ loading detection times """
        detection_time_dict = {}
        wdfs = self.wdfs
        det = self.det_times
        for i in self.narr:
            l = self.labels[i]
            detection_time_dict[l] = dict(start=det[i][0], end=det[i][1], wdf=wdfs[l])
        return detection_time_dict

    def write_json(self, l=''):
        """ write json object to file """
        file_name = self.file_name + l
        if not os.path.isdir(self.shotdir_run):
            os.makedirs(self.shotdir_run)
        file_path = os.path.join(self.shotdir_run, file_name + lp.dt_ex_out)
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

