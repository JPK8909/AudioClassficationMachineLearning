#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:21:50 2018

@author: jin
"""

import os
import os.path
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import librosa

DATA_PATH = '/Users/jin/Downloads/Data/'

def conv_wav_to_mfcc(path, file_full_names, max_length=15):
    wave, sr = librosa.load(path + file_full_names, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    if (max_length > mfcc.shape[1]):
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    else:
        mfcc = mfcc[:, :max_length]
    
    return mfcc

def make_npy_file(full_file_names):
    mfcc_vectors = []
    for i in range(len(full_file_names)):
        print(i)
        mfcc = conv_wav_to_mfcc(DATA_PATH, full_file_names[i])
        mfcc_vectors.append(mfcc)      
    np.save('mfcc_version.npy', mfcc_vectors)
    

def main():
    
    file_names = []
    full_file_names = []
    for filename in os.listdir(DATA_PATH):
        full_file_names.append(filename)
        name = os.path.splitext(filename)[0]
        file_names.append(name)
    full_file_names.pop(1284)
    
    df = pd.Series(file_names)
    df = pd.DataFrame(df.str.split('-',1).tolist(), columns = ['Subclass', 'Target'])
    df = df.dropna(axis = 0)
    
    
    if os.path.isfile('mfcc_version.npy')!=1:
        make_npy_file(full_file_names)

    X = np.load('mfcc_version.npy')
    y = df.iloc[:,1:2].values
    y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=0, shuffle=True)
    
    X_train = X_train.reshape(X_train.shape[0], 20, 15, 1)
    X_test = X_test.reshape(X_test.shape[0], 20, 15, 1)
    
    y_train_category = to_categorical(y_train)
    y_test_category = to_categorical(y_test)
    
    model = Sequential()
    
    model.add(Conv2D(36, kernel_size=(2, 2), activation='relu', input_shape=(20, 15, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(132, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    
    model.fit(X_train, y_train_category, batch_size=150, epochs=200, verbose=1, validation_data=(X_test, y_test_category))
    
if __name__ == '__main__':
    main()























