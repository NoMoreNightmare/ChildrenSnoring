# -*- coding:utf-8 -*-

from configparser import ConfigParser
from time import sleep

import librosa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers
from keras import layers
import keras
from keras.models import Sequential
import warnings

global csv
default_csv="dataset.csv"
warnings.filterwarnings('ignore')


def read_all_file(record_files):
    if os.path.isdir(record_files):
        files = os.listdir(record_files)
        for file in files:
            record_path = record_files + "/" + file
            if os.path.isdir(record_path):
                read_all_file(record_path)
            else:
                # print(record_path)
                list = file.split('.')
                if list[len(list) - 1] == "pk":
                    continue
                else:
                    feature_extract(record_path)
    else:
        # print(record_files)
        feature_extract(record_files)


def feature_extract(audio):
    # for g in genres:
    if True:
        # for filename in os.listdir(f'./drive/My Drive/genres/{g}'):
        songname = audio
        strs = songname.split('/')
        song = strs.pop(len(strs) - 1)
        y, sr = librosa.load(songname, mono=True, duration=30)
        rmse = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{song} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        csv_file = open(default_csv, 'a', newline='')
        with csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(to_append.split())


if __name__ == "__main__":
    # read property file
    config = ConfigParser()
    config.read('property.cfg')
    # get the path of mp3 file
    # mp3 = config.get("mp3files", "mp3file1")
    mp3files = config.options("mp3files")
    sleep(1)

    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    csv_file = open(default_csv, 'w', newline='')
    with csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

    for i in mp3files:
        record_files = config.get("mp3files", i)
        read_all_file(record_files)

    data = pd.read_csv(default_csv)
    data.head()  # Dropping unneccesary columns
    data = data.drop(['filename'], axis=1)  # Encoding the Labels
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)  # Scaling the Feature columns
    print(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))  # Dividing data into training and Testing set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    classifier = model.fit(X_train, y_train, epochs=100, batch_size=128)


def model_create(property, option):
    # read property file
    config = ConfigParser()
    config.read(property)
    # get the path of mp3 file
    # mp3 = config.get("mp3files", "mp3file1")
    mp3files = config.options(option)
    sleep(1)
    print("file name       " + "均方根能量        " + "过零率         " + "谱质心          ", end="")
    print("mfcc         " + "频谱平坦度          " + "频谱通量         " + "基音频率       ", end="")
    print("响度          " + "尖锐度")

    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    csv_file = open(default_csv, 'a', newline='')
    with csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

    for i in mp3files:
        record_files = config.get(option, i)
        read_all_file(record_files)

    data = pd.read_csv(default_csv)
    data.head()  # Dropping unneccesary columns
    data = data.drop(['filename'], axis=1)  # Encoding the Labels
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)  # Scaling the Feature
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))  # Dividing data into training and Testing set
    # return X
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    print(X_train)

    return X_train, X_test, y_train, y_test



def own_data(record_files,head):

    global default_csv
    default_csv="temp.csv"

    if head==0:
        header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label'
        header = header.split()

        csv_file = open(default_csv, 'w', newline='', encoding='utf8', errors='ignore')
        with csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

    read_all_file(record_files)

    # data = pd.read_csv(default_csv)
    # data.head()  # Dropping unneccesary columns
    # data = data.drop(['filename'], axis=1)  # Encoding the Labels
    # genre_list = data.iloc[:, -1]
    # scaler = StandardScaler()
    # X=np.array(data.iloc[:, :-1], dtype=float) # Dividing data into training and Testing set
    # return X
