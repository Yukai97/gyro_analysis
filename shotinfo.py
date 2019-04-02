import json
from gyro_analysis.local_path import *


class ShotInfo:
    def __init__(self, name, run_number):
        self.name = name
        self.ext = '.hdr'
        self.run_number = run_number

        infodir_run = os.path.join(infodir, run_number)
        file_path = os.path.join(infodir_run, self.name + self.ext)
        try:
            read_header = open(file_path, 'r')
            self.exists = True
        except FileNotFoundError:
            self.exists = False
        else:
            header = json.load(read_header)
            self.waveform_config = header['waveform_config']
            self.shot_number = header['shot_number']
            self.he_angle = header['he_angle']
            self.ne_angle = header['ne_angle']
            self.xe_angle = header['xe_angle']
            self.timestamp = header['timestamp']
            read_header.close()

