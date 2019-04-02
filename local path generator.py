import json
import os

homedir = r"C:\Users\Romalis Group\Desktop\New Data"
rawdir = os.path.join(homedir, 'Raw data')
infodir = os.path.join(homedir, 'Info data')
scdir = os.path.join(homedir, 'SC data')
shotdir = os.path.join(homedir, 'Shot data')

path_dict = {'homedir': homedir, 'rawdir': rawdir, 'infodir': infodir, 'scdir': scdir, 'shotdir': shotdir}
json_output = json.dumps(path_dict)
f = open('local_path.json', 'w')
f.write(json_output)
f.close()
