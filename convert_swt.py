#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir
import shlex, subprocess


# In[2]:


import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join


# In[3]:


import json


# In[4]:


import numpy as np
import math
import shutil
import os.path


# In[5]:


def get_swt(file_name,data_path,swt_path):
    if file_name[-4:] not in ['.png', '.jpg']:
        return

    #modify the path to where your stroke-width-transformation package is
    command_line = 'python ./stroke-width-transform/main.py ' + data_path + file_name + ' ' + swt_path + file_name      
    print(command_line)
    args = command_line.split(' ')
    print(args[-1][:-2])    
    p = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    print(p.communicate())

from joblib import Parallel, delayed


# In[13]:


def load_files(img_dir,out_dir):
    
    img_dir = img_dir[2:]
    out_dir = out_dir[3:]
    print(img_dir)

    print(out_dir)
    
    file_list = listdir(img_dir)

    Parallel(n_jobs=32)(delayed(get_swt)(file_list[i],img_dir,out_dir) for i in range(len(file_list)))
    
#load_files( input folder, output folder)
load_files( '  ./data/pipeline_data/Open_Access/Bar/', '   ./Open_Access_swt/')




