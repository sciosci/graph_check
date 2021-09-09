import os
from os import listdir
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
import subprocess

def get_role(file_name,data_path,labeled_data):
    if file_name[-4:] != '.png':
        return

    my_file = Path(labeled_data + file_name[:-4] + '-texts-prob.csv')
    if my_file.exists():
        t2 = pd.read_csv(labeled_data + file_name[:-4] + '-texts-prob.csv')
        if t2.shape[0] <2:
            return
    else:
        return

    command_line = 'python ./Graphical_Integrity_Issues/rev/run_text_role_classifier.py single '+ data_path + file_name
    args = command_line.split(' ')
    p = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print(p.communicate())
    t1 = pd.read_csv(labeled_data+file_name[:-4]+'-pred1-texts.csv')
    t_all = t1.merge(t2,on='id')
    t_all.to_csv(labeled_data+file_name[:-4]+'-texts-all.csv')


labeled_data = './Graphical_Integrity_Issues/Open_Access_swt/'
data_path = './Graphical_Integrity_Issues/Open_Access_swt/'


file_names = listdir(labeled_data)

Parallel(n_jobs=4)(delayed(get_role)(file_names[i],data_path,labeled_data) for i in range(len(file_names)))
