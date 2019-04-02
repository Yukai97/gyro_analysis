import os
import json
import sys
import pkg_resources


class ShotInfo:
    def __init__(self, name, run_number):
        self.name = name
        self.ext = '.hdr'
        self.run_number = run_number
        try:
            f = open(pkg_resources.resource_filename(__name__, 'local_path.json'), 'r')
        except FileNotFoundError:
            print('Do not find local_path.json file! Please use "local path generator.py" to generate this file')
            sys.exit()
        else:
            local_path = json.load(f)
            self.homedir = local_path['homedir']
            self.rawdir = local_path['rawdir']
            self.infodir = local_path['infodir']
            self.scdir = local_path['scdir']
            self.shotdir = local_path['shotdir']
            f.close()

        infodir_run = os.path.join(self.infodir, run_number)
        file_path = os.path.join(infodir_run, self.name + self.ext)
        try:
            read_header = open(file_path, 'r')
        except FileNotFoundError:
            print('Do not find header file for' + name)
            print('Please check if this header file exists and if your local paths in local_path.json are right.')
        else:
            header = json.load(read_header)
            self.waveform_config = header['waveform_config']
            self.shot_number = header['shot_number']
            self.he_angle = header['he_angle']
            self.ne_angle = header['ne_angle']
            self.xe_angle = header['xe_angle']
            self.timestamp = header['timestamp']
            read_header.close()

