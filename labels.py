# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os, sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
import glob
import os



Arousal= "/Users/shrutikshirsagar/Downloads/Arousal/"

print(os.listdir(Arousal))
Valence = "/Users/shrutikshirsagar/Downloads/Valence/"
path2 = "/Users/shrutikshirsagar/Downloads/labels/"
if not os.path.exists(path2):
    os.makedirs(path2)
for file in os.listdir(Arousal):
    print('file', file)

    for j in os.listdir(Valence):
        print(j)
        if file == j:
            print('processing', file)
            df = pd.read_csv(os.path.join(Arousal,file), delimiter=',')
            print('Arousal shape', df.shape)
    
            DS = pd.read_csv(os.path.join(Valence, j), sep=',')
            print('valence shape', DS.shape)
            
            DS = DS.values
            print(DS.shape)
            df = df.values
            print(df.shape)
            DS1 = DS[:,-1].astype(float)
            DS1 = DS1.reshape((DS1.shape[0], 1))
            print('DS1 shape', DS1.shape)
            df_1=pd.DataFrame(np.concatenate([df,DS1],1))
            df_1.to_csv((os.path.join(path2, file)), index = False,header = None)
        
  
                   
