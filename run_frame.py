import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from gyro_analysis.local_path import paths as pt
from gyro_analysis.shot_frame import ShotFrame
shotdir = pt.shotdir

class RunDataFrame:

    def __init__(self, run_number=None, shot_frame_list=None, amp_lb=0, amp_ub=5):
        self.amp_lb = amp_lb
        self.amp_ub = amp_ub
        if run_number:
            self.shot_frame_list = self.build_shot_frame_list(run_number)
        else:
            self.shot_frame_list = shot_frame_list
        self.run_frame = self.build_run_frame()

    def build_shot_frame_list(self, run_number):
        if type(run_number) == str or type(run_number) == int:
            run_number = str(int(run_number)).zfill(4)
            run_number = [run_number]
        shot_frame_list = []
        for i in run_number:
            shotdir_run = os.path.join(shotdir, i)
            file_list = sorted(os.listdir(shotdir_run))
            for file in file_list:
                shot_number = file.split('.')[0].split('_')[1]
                shot_frame_list.append(ShotFrame(i, shot_number, self.amp_lb, self.amp_ub))
        return shot_frame_list

    def build_run_frame(self):
        shot_frames = [self.shot_frame_list[i].shot_frame for i in range(len(self.shot_frame_list))]
        run_frame = pd.concat(shot_frames)
        run_frame = run_frame.reset_index()
        run_frame = run_frame.set_index(['run_number', 'shot_number', 'label'])
        return run_frame