#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:02:34 2020

@author: shrutikshirsagar
"""
##replace all filename
import os
paths = (os.path.join(root, filename)
        for root, _, filenames in os.walk('//home/shrutikshirsagar/Downloads/noise_add/rev_new')
        for filename in filenames)

for path in paths:
    # the '#' in the example below will be replaced by the '-' in the filenames in the directory
    newname = path.replace('_0.83.wav', '.wav')
    if newname != path:
        os.rename(path, newname) 


##remove files from folders
import os
paths = (os.path.join(root, filename)
        for root, _, filenames in os.walk('/media/shrutikshirsagar/Data/SEWA_clean')
        for filename in filenames)

for path in paths:
    os.remove(path)

##copy clean file to all subdirectories in a  folder

import os
import shutil
import glob

def main():
    
    src = "/home/shrutikshirsagar/Downloads/noise_add/clean"
    dst = "/media/shrutikshirsagar/Data/SEWA_clean"
 
   
    for root, dirnames, filenames in os.walk(dst):
        print(root)
        for filePath in glob.glob(src + '/*.wav'):
            # Move each file to destination Directory
            print(filePath)
            #print(os.path.join(filepath))
           # print(os.path.join(root, ))
            
            shutil.copy(filePath, root)
              

if __name__ == '__main__':
    main()



##SEGAN data prepapartion- filename change according to foldername
##this script rename all files in subdirectory with that directory name


import os
path =  "/media/shrutikshirsagar/Data/SEWA_clean"

for root, dirname, filename in os.walk(path):
    s = os.path.basename(root)
    #print('s',s)
    
    for name in filename:
        old_name = os.path.join(root, name)
        #print('oldname',old_name)
       
        newname = name.replace('Devel' , s + '_Devel' )
        #print(newname)
        
        new_name1 = os.path.join(root, newname)
        #print(new_name1)
        os.rename(old_name,new_name1)
        

##move to one folder
import os
import shutil
import glob

def main():
    
    src =   "/home/shrutikshirsagar/Downloads/noise_add/SEWA_training_data"
    dst =  "/media/shrutikshirsagar/Data/segan_pytorch-master/data/noisy_trainset"
        
    if not os.path.exists(dst): 
        os.makedirs(dst)
   
    for root, dirnames, filenames in os.walk(src):
        for filename in filenames:
            a = os.path.join(root, filename)
            
            
            shutil.move(a, dst)
              

if __name__ == '__main__':
    main()




