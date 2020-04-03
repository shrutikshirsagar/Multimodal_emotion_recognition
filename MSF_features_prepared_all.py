import os, sys
import pandas as pd

import librosa    
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from pathlib import Path
import glob
import os

pathin = "/media/shrutikshirsagar/Data/Multimodal_journa/RECOLA_audio"
pathout = "/media/shrutikshirsagar/Data/Multimodal_journa/MSF_prepared_RECOLA"
path2 = "/media/shrutikshirsagar/Data/Multimodal_journa/ratings_gold_standard/arousal/"
dirFiles2 = os.listdir(path2) #list of directory files
dirFiles2.sort()
print(dirFiles2)



def filematch(pathin, pathout):
    dirs = os.listdir("%s/%s"%(pathin, dirname))
    print('directory name', dirs)
    dirFiles1 = dirs #list of directory files
    dirFiles1.sort()
    print(dirFiles1)
    if not os.path.exists("%s/%s" % (pathout, dirname)):
    
        os.makedirs("%s/%s" % (pathout, dirname))
    for file in dirs:
        print('file', file)
        if file.endswith('.csv'):
            print('processing new %s/%s/%s' % (pathin, dirname, file))
            a = os.path.join(pathin, dirname, file)
            print('processing out %s/%s/%s' % (pathout, dirname, file))
            b = os.path.join(pathout, dirname, file)
            print('a', a)
            print('b', b)
           # for i in dirFiles1:
                #print(i)
            for j in dirFiles2:
                print(j)
                if file == j:
                    df = pd.read_csv(os.path.join(a), delimiter=',', header = None)
                    print(df.shape)

                    DS = pd.read_csv(os.path.join(path2, j), sep=',')
                    print(DS.shape)
                    if df.shape[0] > DS.shape[0]:
                        n = df.shape[0] - DS.shape[0]
                        print(n)
                        df.drop(df.tail(n).index,inplace=True)

                        print(df.shape)
                        DS = DS.values
                        names, times, DS = DS[:,0:1], DS[:,1:2], DS[1:,2:].astype(float)
                        df_1=pd.DataFrame(np.concatenate([names, times, df],1))
                        df_1.to_csv(b, index = False,header = None)
                    else:
                        n = DS.shape[0] - df.shape[0]
                        print(n)
                        DS.drop(DS.tail(n).index,inplace=True)

                        print(DS.shape)
                        DS = DS.values
                        names, times, DS = DS[:,0:1], DS[:,1:2], DS[1:,2:].astype(float)
                        df_1=pd.DataFrame(np.concatenate([names, times, df],1))
                        df_1.to_csv(b, index = False,header = None)
   
for dirname in os.listdir(pathin):
    print('directname', dirname)
    print('path in', os.listdir(pathin))
    print("Processing: %s/%s" % (pathin, dirname))
    #if os.path.isdir("%s/%s" % (pathin, dirname)):
    #pathout = "%s/%s" % (pathout, dirname)
    
    filematch(pathin, pathout)
        #if not os.path.exists(pathout):
            
                           
