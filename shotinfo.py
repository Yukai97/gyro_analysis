import json
import time
from gyro_analysis.local_path import *


class ShotInfo:
    """Contain necessary information for a shot

    Args:
        run_number(str): specify the specific run
        name(str): specify the specific shot
    """
    def __init__(self, run_number, name):
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
            self.timestamp = self.get_timestamp()
        else:
            header = json.load(read_header)
            self.waveform_config = header['waveform_config']
            self.shot_number = header['shot_number']
            self.he_angle = header['he_angle']
            self.ne_angle = header['ne_angle']
            self.xe_angle = header['xe_angle']
            self.timestamp = header['timestamp']
            self.cycle_number = header['cycle_number']
            self.sequence_var = header['sequence_var']
            read_header.close()

    def get_timestamp(self):
        rawdir_run = os.path.join(rawdir, self.run_number)
        full_file = os.path.join(rawdir_run, self.name + '.rdt')
        t = os.path.getctime(full_file)
        tmptime = time.localtime(t)
        return time.strftime('%Y-%m-%d', tmptime)


