#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:00:59 2020

@author: shrutikshirsagar
"""


from __future__ import print_function
import os
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Masking, LSTM, TimeDistributed, Bidirectional
from keras.optimizers import RMSprop

from calc_scores import calc_scores

import pandas as pd
from numpy.random import seed
from tensorflow import set_random_seed


# Helper functions
def get_num_lines(filename, skip_header=False):
    with open(filename, 'r',encoding='ISO-8859-1') as file:
        c = 0
        if skip_header:
            c = -1
        for line in file:
            c += 1
    return c

def get_num_columns(filename, delim=';', skip_header=False):
    # Returns the number of columns in a csv file
    # First two columns must be 'instance name' and 'timestamp' and are not considered in the output
    with open(filename, 'r',encoding='ISO-8859-1') as file:
        if skip_header:
            next(file)
        line = next(file)
        offset1 = line.find(delim)+1
        offset2 = line[offset1:].find(delim)+1+offset1
        cols = np.fromstring(line[offset2:], dtype=float, sep=delim)
      
    return len(cols)

def read_csv(filename, delim=';', skip_header=False):
    # Returns the content of a csv file (delimiter delim, default: ';')
    # First two columns must be 'instance name' and 'timestamp' and are not considered in the output, header is skipped if skip_header=True
    num_lines = get_num_lines(filename, skip_header)
   
    data = np.empty((num_lines,get_num_columns(filename,delim,skip_header)), float)
 
    with open(filename, 'r',encoding='ISO-8859-1') as file:
        if skip_header:
            next(file)
        c = 0
        for line in file:
            offset1 = line.find(delim)+1
            offset2 = line[offset1:].find(delim)+1+offset1
            data[c,:] = np.fromstring(line[offset2:], dtype=float, sep=delim)
            c += 1
    return data


def load_features(path_features='../Prosody/', partition='Train_DE', num_inst=34, max_seq_len=1768):
    skip_header  = False  # AVEC 2018 XBOW feature files
    num_features = get_num_columns(path_features + '/' + partition + '_01.csv', delim=';', skip_header=skip_header)  # check first 
    
    features = np.empty((num_inst, max_seq_len, num_features))
    for n in range(0, num_inst):
        F = read_csv(path_features + '/' + partition + '_' + str(n+1).zfill(2) + '.csv', delim=';', skip_header=skip_header)
        print(F)
        print(path_features)
        if F.shape[0]>max_seq_len:
            F = F[:max_seq_len,:]  # cropping
        features[n,:,:] = np.concatenate((F, np.zeros((max_seq_len - F.shape[0], num_features))))  # zero padding
    
    return features


def load_labels(path_labels='../labels/', partition='Train_DE', num_inst=34, max_seq_len=1768, targets=[0,1,2]):
    # targets=[0,1,2]: 0: arousal, 1: valence, 2: liking/likability
    skip_header = False  # AVEC 2018 XBOW labels files
    num_labels  = len(targets)
    
    labels_original = []
    labels_padded   = []
    
    for t in targets:
        labels_original_t = []
        labels_padded_t   = np.empty((num_inst, max_seq_len, 1))
        
        for n in range(0, num_inst):
            yn = read_csv(path_labels + partition + '_' + str(n+1).zfill(2) + '.csv', skip_header=skip_header)
            yn = yn[:,t].reshape((yn.shape[0], 1))  # select only target dimension and reshape to 2D array
            # original length
            labels_original_t.append(yn)
            # padded to maximum length
            if yn.shape[0] > max_seq_len:
                yn = yn[:max_seq_len]
            #print(yn.shape)
            labels_padded_t[n,:,:] = np.concatenate((yn, np.zeros((max_seq_len - yn.shape[0], 1))))  # zero padding        
            #print(labels_padded_t.shape)
        labels_original.append(labels_original_t)
        labels_padded.append(labels_padded_t)
    
    return labels_original, labels_padded


def load_CES_data(path, use_audio=True, use_visual=False, use_linguistic=False, targets=[0,1,2]):
    num_train_DE = 34  # number of recordings
    num_devel_DE = 14

    
    max_seq_len = 1768  # maximum number of labels
    
    # Initialise numpy arrays
    train_DE_x = np.empty((num_train_DE, max_seq_len, 0))
    devel_DE_x = np.empty((num_devel_DE, max_seq_len, 0))
    
    if use_audio:
        train_DE_x = np.concatenate( (train_DE_x, load_features(path_features= path, partition='Train_DE', num_inst=num_train_DE, max_seq_len=max_seq_len) ), axis=2)
        devel_DE_x = np.concatenate( (devel_DE_x, load_features(path_features= path, partition='Devel_DE', num_inst=num_devel_DE, max_seq_len=max_seq_len) ), axis=2)
        #test_DE_x  = np.concatenate( (test_DE_x,  load_features(path_features='../xbow_prosody77/', partition='Test_DE',  num_inst=num_test_DE,  max_seq_len=max_seq_len) ), axis=2)
        #test_HU_x  = np.concatenate( (test_HU_x,  load_features(path_features='../xbow_prosody77/', partition='Test_HU',  num_inst=num_test_HU,  max_seq_len=max_seq_len) ), axis=2)
    if use_visual:
        train_DE_x = np.concatenate( (train_DE_x, load_features(path_features='../Visual_features_500_xbow/', partition='Train_DE', num_inst=num_train_DE, max_seq_len=max_seq_len) ), axis=2)
        devel_DE_x = np.concatenate( (devel_DE_x, load_features(path_features='../Visual_features_500_xbow/', partition='Devel_DE', num_inst=num_devel_DE, max_seq_len=max_seq_len) ), axis=2)
        #test_DE_x  = np.concatenate( (test_DE_x,  load_features(path_features='../xvod_prosody77/', partition='Test_DE',  num_inst=num_test_DE,  max_seq_len=max_seq_len) ), axis=2)
        #test_HU_x  = np.concatenate( (test_HU_x,  load_features(path_features='../xvod_prosody77/', partition='Test_HU',  num_inst=num_test_HU,  max_seq_len=max_seq_len) ), axis=2)
    if use_linguistic:
        train_DE_x = np.concatenate( (train_DE_x, load_features(path_features='../text_features_xbow_6s/', partition='Train_DE', num_inst=num_train_DE, max_seq_len=max_seq_len) ), axis=2)
        devel_DE_x = np.concatenate( (devel_DE_x, load_features(path_features='../text_features_xbow_6s/', partition='Devel_DE', num_inst=num_devel_DE, max_seq_len=max_seq_len) ), axis=2)
        #test_DE_x  = np.concatenate( (test_DE_x,  load_features(path_features='../linguistic_features_xbow/', partition='Test_DE',  num_inst=num_test_DE,  max_seq_len=max_seq_len) ), axis=2)
        #test_HU_x  = np.concatenate( (test_HU_x,  load_features(path_features='../linguistic_features_xbow/', partition='Test_HU',  num_inst=num_test_HU,  max_seq_len=max_seq_len) ), axis=2)
    
    _                       , train_DE_y = load_labels(path_labels='../labels/', partition='Train_DE', num_inst=num_train_DE, max_seq_len=max_seq_len, targets=targets)
    devel_DE_labels_original, devel_DE_y = load_labels(path_labels='../labels/', partition='Devel_DE', num_inst=num_devel_DE, max_seq_len=max_seq_len, targets=targets)
    
    return train_DE_x, train_DE_y, devel_DE_x, devel_DE_y, devel_DE_labels_original

def emotion_model(max_seq_len, num_features, learning_rate, num_units_1, num_units_2, bidirectional, dropout, num_targets):
    # Input layer
    inputs = Input(shape=(max_seq_len,num_features))
    
    # Masking zero input - shorter sequences
    net = Masking()(inputs)
    
    # 1st layer
    if bidirectional:
        net = Bidirectional(LSTM( num_units_1, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(net)
    else:
        net = LSTM(num_units_1, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(net)
    
    # 2nd layer
    if bidirectional:
        net = Bidirectional(LSTM( num_units_2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout ))(net)
    else:
        net = LSTM(num_units_2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(net)
    
    # Output layer
    outputs = []
    out1 = TimeDistributed(Dense(1))(net)  # linear activation
    outputs.append(out1)
    if num_targets>=2:
        out2 = TimeDistributed(Dense(1))(net)  # linear activation
        outputs.append(out2)
    if num_targets==3:
        out3 = TimeDistributed(Dense(1))(net)  # linear activation
        outputs.append(out3)
    
    # Create and compile model
    rmsprop = RMSprop(lr=learning_rate)
    model   = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=rmsprop, loss=ccc_loss)  # CCC-based loss function
    return model



def main():
    
    
    
    pathin = "/Users/shrutikshirsagar/Documents/LSTM_experiments/data"
    output_fin = np.empty((0,4))
    for dirname in os.listdir(pathin):
        
    
        new_path = os.path.join(pathin, dirname)
        #print('new_path', new_path )
      
        folder_audio_features =  new_path 
        #print('folder_audio', folder_audio_features)
        
        
        
        path_output = 'predictions/'  # To store the predictions on the test partitions
        
        # Modalities
        use_audio      = True
        use_visual     = False
        use_linguistic = False
        path = folder_audio_features
        print('path',path)
        
        # Neural net parameters
        batch_size    = 34       # Full-batch: 34 sequences
        learning_rate = 0.001    # default is 0.001
        num_iter      = 3       # Number of Iterations
        num_units_1   = 64       # Number of LSTM units in LSTM layer 2
        num_units_2   = 32       # Number of LSTM units in LSTM layer 2
        bidirectional = False    # True/False
        dropout       = 0.1      # Dropout
        
        # Targets
        targets       = [0,1,2]  # List of targets: 0=arousal, 1=valence, 2=liking
        shift_sec     = 2.0 # Shift of annotations for training (in seconds)
        
        ##
        target_names = {0: 'arousal', 1: 'valence', 2: 'liking'}
        inst_per_sec = 10  # 100ms hop size
        
        # Set seeds to make results reproducible 
        # (Note: Results might be different from those reported by the Organisers as seeds also training depends on hardware!)
        seed(1)
        set_random_seed(2)
        
        num_targets = len(targets)  # same for all Y
        
        shift = int(np.round(shift_sec*inst_per_sec))  
        
        # Load AVEC2018-CES data
        print('Loading data ...')
        train_x, train_y, devel_x, devel_y, devel_labels_original = load_CES_data(path, use_audio, use_visual, use_linguistic, targets)
        num_train    = train_x.shape[0]
        num_devel    = devel_x.shape[0]
       
        max_seq_len  = train_x.shape[1]  # same for all partitions
        num_features = train_x.shape[2]
        print(' ... done')
    
      
        
        # Shift labels to compensate annotation delay
        print('Shifting labels to the front for ' + str(shift_sec) + ' seconds ...')
        for t in range(0, num_targets):
            train_y[t] = shift_labels_to_front(train_y[t], shift)
            devel_y[t] = shift_labels_to_front(devel_y[t], shift)
        print(' ... done')
        
        # Create model
        model = emotion_model(max_seq_len, num_features, learning_rate, num_units_1, num_units_2, bidirectional, dropout, num_targets)
        # print(model.summary())
        
        # Train and evaluate model
        ccc_devel_best = np.zeros(num_targets)
        print('ccc', ccc_devel_best)
       
       
        for iteration in range(num_iter):
        
            print('Iteration: ' + str(iteration))
           
            model.fit(train_x, train_y, batch_size=batch_size, epochs=1)  # Evaluate after each epoch
            
            # Evaluate on development partition
            ccc_iter = evaluate_devel(model, devel_x, devel_labels_original, shift, targets)
            
            # Print results
            print('CCC Devel (', end='')
            for t in range(0, num_targets):
                print(target_names[targets[t]] + ',', end='')
            print('): ' + str(np.round(ccc_iter*1000)/1000))
            
            # Get predictions on test (and shift back) if CCC on Devel improved
            for t in range(0, num_targets):
                if ccc_iter[t] > ccc_devel_best[t]:
                    ccc_devel_best[t] = ccc_iter[t]
        print('CCC Devel best (', end='')
        for t in range(0, num_targets):
            print(target_names[targets[t]] + ',', end='')
        print('): ' + str(np.round(ccc_devel_best*1000)/1000))
        
        folder_name = folder_audio_features
        out_vec=np.hstack((folder_name, (np.round(ccc_devel_best*1000)/1000)))
        print
            
        output_fin=np.vstack((output_fin,out_vec))
    df=pd.DataFrame(output_fin)
    df.to_csv('output_df7.csv', index=None)
    

def evaluate_devel(model, devel_x, label_devel, shift, targets):
   
    num_targets = len(targets)
    CCC_devel   = np.zeros(num_targets)
    # Get predictions
    pred_devel = model.predict(devel_x)
    # In case of a single target, model.predict() does not return a list, which is required
    if num_targets==1:
        pred_devel = [pred_devel]    
    for t in range(0,num_targets):
        # Shift predictions back in time (delay)
        pred_devel[t] = shift_labels_to_back(pred_devel[t], shift)
        CCC_devel[t]  = evaluate_partition(pred_devel[t], label_devel[t])
    return CCC_devel


def evaluate_partition(pred, gold):
    # pred: np.array (num_seq, max_seq_len, 1)
    # gold: list (num_seq) - np.arrays (len_original, 1)
    pred_all = np.array([])
    gold_all = np.array([])
    for n in range(0, len(gold)):
        # cropping to length of original sequence
        len_original = len(gold[n])
        pred_n = pred[n,:len_original,0]
        # global concatenation - evaluation
        pred_all = np.append(pred_all, pred_n.flatten())
        gold_all = np.append(gold_all, gold[n].flatten())
    ccc, _, _ = calc_scores(gold_all,pred_all)
    return ccc

def shift_labels_to_front(labels, shift=0):
    labels = np.concatenate((labels[:,shift:,:], np.zeros((labels.shape[0],shift,labels.shape[2]))), axis=1)
    return labels


def shift_labels_to_back(labels, shift=0):
    labels = np.concatenate((np.zeros((labels.shape[0],shift,labels.shape[2])), labels[:,:labels.shape[1]-shift,:]), axis=1)
    return labels


def ccc_loss(gold, pred):  # Concordance correlation coefficient (CCC)-based loss function - using non-inductive statistics
    # input (num_batches, seq_len, 1)
    gold       = K.squeeze(gold, axis=-1)
    pred       = K.squeeze(pred, axis=-1)
    gold_mean  = K.mean(gold, axis=-1, keepdims=True)
    pred_mean  = K.mean(pred, axis=-1, keepdims=True)
    covariance = (gold-gold_mean)*(pred-pred_mean)
    gold_var   = K.mean(K.square(gold-gold_mean), axis=-1, keepdims=True)
    pred_var   = K.mean(K.square(pred-pred_mean), axis=-1, keepdims=True)
    ccc        = K.constant(2.) * covariance / (gold_var + pred_var + K.square(gold_mean - pred_mean) + K.common.epsilon())
    ccc_loss   = K.constant(1.) - ccc
    return ccc_loss










if __name__ == '__main__':
    main()
#plot baseline and prediction
#plt.plot(iteration, ccc_Devel)
