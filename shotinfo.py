import json
import time
from gyro_analysis.local_path import paths as lp
from gyro_analysis.local_path import extensions as ext
import os


class ShotInfo:
    """Contain necessary information for a shot

    Args:
        run_number(str): specify the specific run
        name(str): specify the specific shot
    """
    def __init__(self, run_number, shot_number):
        self.run_number = '{:04d}'.format(int(run_number))
        self.shot_number = '{:03d}'.format(int(shot_number))
        self.file_name = '_'.join([self.run_number, self.shot_number])
        file_path = os.path.join(lp.infodir, self.run_number, self.file_name + ext.shot_header)
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
        full_file = os.path.join(lp.rawdir, self.run_number, self.file_name + ext.rd_in)
        t = os.path.getctime(full_file)
        tmptime = time.localtime(t)
        return time.strftime('%Y-%m-%d', tmptime)


