#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 22:49:45 2020

@author: shrutikshirsagar
"""


from glob import glob
import numpy as np
import os, time
import pandas as pd




data_path =  "/Users/shrutikshirsagar/Downloads/Arousal/"

output_file ="/Users/shrutikshirsagar/Downloads/labels/"

ref_path =  "/Users/shrutikshirsagar/Downloads/Valence/"

files = glob(os.path.join(data_path, '*.csv'))
print('total files', files)

for filename in files:
    print('filename', filename)
    print('basename', os.path.basename(filename))
    ref_file = os.path.join(ref_path, os.path.basename(filename))
    print('ref_file', ref_file)
    print('processing', filename)
    df = pd.read_csv(filename, delimiter=',')
    print('Arousal shape', df.shape)

    DS = pd.read_csv(ref_file, sep=',')
    print('valence shape', DS.shape)
    
    DS = DS.values
    print(DS.shape)
    df = df.values
    print(df.shape)
    DS1 = DS[:,-1].astype(float)
    DS1 = DS1.reshape((DS1.shape[0], 1))
    print('DS1 shape', DS1.shape)
    df_1=pd.DataFrame(np.concatenate([df,DS1],1))
    df_1.to_csv(output_file+os.path.basename(filename), index = False,header = None)
    
    