import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd


class DataPackage:

    def __init__(self, shot_series_list):
        self.shot_series_list = shot_series_list
        self.data_frame = self.build_data_frame()

    def build_data_frame(self):
        ss_list = self.shot_series_list
        for i in range(len(ss_list)):
            for l in ss_list[i].labels:
                sr = pd.concat([ss_list[i].shot_series[l]])
