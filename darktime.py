import numpy as np
from string import ascii_lowercase
from matplotlib import pyplot as plt
import os
import json_tricks as jsont

from gyro_analysis.rawdata import RawData
from gyro_analysis.rawfitter import RawFitter
from gyro_analysis.scfitter import SClist
from gyro_analysis.scfitter import ShdReader
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
        self.write_dir = os.path.join(lp.shotdir, self.run_number)
        self.ext = 'sho'
        self.n_det = len(self.det_times)
        self.narr = np.arange(self.n_det)
        self.labels = ascii_lowercase[:self.n_det]
        self.detection_time_dict = self.make_detection_time_dict()
        self.dark_time_dict = self.make_dark_time_dict()  # modified by calc_dark_time
        self.ok = ['y', 'Y']
        self.file_name = '_'.join([self.run_number, self.shot_number])
        self.check_init()
        self.res_max = np.zeros(self.n_det)
        #  self.process_rdt_to_sco(ddict)
        if check_raw:
            self.check_raw_res()
        # self.process_sco_to_shd()
        if check_phases:
            self.check_phase_res()
        self.shds = {l: ShdReader(self.run_number, self.shot_number, l) for l in self.labels}
        self.dark_time_dict = self.calc_dark_time()
        self.output_dict = self.make_output()
        # self.process_shd_to_dto()

    def make_dark_time_dict(self):
        """ dict of dark-time segments """
        dark_time_dict = {}
        for i in self.narr[:-1]:
            dt_label = self.labels[i] + self.labels[i + 1]
            time_diff = self.det_times[i + 1][0] - self.det_times[i][1]
            dark_time_dict[dt_label] = dict(time_diff=time_diff,
                                            time_start=self.det_times[i][1],
                                            time_end=self.det_times[i + 1][0],
                                            phase_diff={},
                                            phase_diff_err={},
                                            freq={},
                                            freq_err={})
        return dark_time_dict

    def make_detection_time_dict(self):
        """ dict of detection-time segments """
        dtd = {}
        det = self.det_times
        for i in self.narr:
            l = self.labels[i]
            dtd[l] = dict(start=det[i][0], end=det[i][1])
        return dtd

    def check_init(self):
        """ Validate parameters """
        print('check_init not yet implemented')
        return

    def check_raw_res(self):
        """ Display max residuals from each raw data fit """
        self.plot_res_max()
        res_ok = input("Are all detection times okay, Y or N? [Y] ") or 'Y'
        if not res_ok in self.ok:
            raise ValueError('Data glitch in {}; code not yet able to handle glitches.')
        return

    def check_phase_res(self):
        """ Show residuals from phase fits """
        print('check_phase_res not yet implemented')
        return

    def process_rdt_to_sco(self):
        """ run fitters on each detection time and write output files """
        fp = self.parameters
        for i in self.narr:
            fp['roi'] = self.det_times[i]
            rd = RawData(self.run_number, self.file_name, fp)
            rf = RawFitter(rd)
            self.res_max[i] = rf.res_max
            rf.process_blocks()
            rf.write_json(l=self.labels[i])
        return

    def process_sco_to_shd(self):
        """ fit sco files to find frequencies and phases during each detection period """
        for l in self.labels:
            sc = SClist(self.run_number, self.shot_number, l)
            sc.write_json()
        return

    def process_shd_to_dto(self):
        """ determine dark time frequencies; load and record relevant light time values """
        self.write_json()
        return

    def calc_dark_time(self):
        """ determine dark time frequencies """
        shds = self.shds
        od = self.dark_time_dict
        for key in self.dark_time_dict:
            l1 = key[0]
            l2 = key[1]
            for sp in shds[l1].fkeys:
                phdf = shds[l2].phase_start[sp] - shds[l1].phase_end[sp]
                phdf_err = np.sqrt(shds[l2].phase_err[sp]**2 + shds[l1].phase_err[sp]**2)
                td = self.dark_time_dict[key]['time_diff']
                od[key]['phase_diff'][sp] = phdf
                od[key]['phase_diff_err'][sp] = phdf_err
                od[key]['freq'][sp] = phdf/td
                od[key]['freq_err'][sp] = phdf_err/td
        return od

    def load_detection_time(self):
        return

    def make_output(self):
        """ generate output json object """
        out_dict = self.detection_time_dict.copy()
        out_dict.update(self.dark_time_dict)
        return out_dict

    def write_json(self, l=''):
        """ write json object to file """
        file_name = self.file_name + l
        if not os.path.isdir(self.write_dir):
            os.makedirs(self.write_dir)
        file_path = os.path.join(self.write_dir, file_name + self.ext)
        od = self.output_dict
        f = open(file_path, 'w')

        def default_json(o):
            return o.__dict__

        json_output = jsont.dumps(od, default=default_json)
        f.write(json_output)
        f.close()


    def plot_res_max(self):
        fig, ax = plt.subplots()
        ax.plot(self.res_max)
        ax.set_xlabel('Detection Time')
        ax.set_ylabel('Residual')
        ax.set_title('Maximum Residual Seen in Block Fitting')
        plt.show()

