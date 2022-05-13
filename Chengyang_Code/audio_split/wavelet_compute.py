import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
import pywt
import pywt.data
import pandas as pd
from pyAudioAnalysis import audioBasicIO as aIO
import transfer_amplitude as ta


def compute_wpe(signal):
    wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=3)
    #Depending on the frequency band ( freq)Sort
    n = 3
    re = []  #No. n Decomposition factor of all nodes in layer
    for i in [node.path for node in wp.get_level(n, 'freq')]:
        re.append(wp[i].data)
    #No. n Layer Energy Characteristics
    energy = []
    for i in re:
        energy.append(np.sqrt(np.sum(i**2)/i.shape[0]))

    return energy


def compute_wte(signal):
    cA, cD, cD_2, cD_1 = pywt.wavedec(data=signal, wavelet='db1',mode='symmetric', level=3)
    energy = []
    sum3 = np.mean(cD**2)+np.mean(cD_2**2)+np.mean(cD_1**2)
    energy.append(np.mean(cD**2)/sum3 * 100)
    energy.append(np.mean(cD_2**2)/sum3 * 100)
    energy.append(np.mean(cD_1**2)/sum3 * 100)
    energyA = pow(np.linalg.norm(cA, ord=None), 2)
    return energy, energyA


# [fs, s] = aIO.read_audio_file('dropNoice/10_mp3.wav')
# s = aIO.stereo_to_mono(s)
#
# da = ta.reduce_noise(s, fs)
# da = ta.amplitude_normalize(da)
# energy, energyA = compute_wte(da)
# print(energy)
# print(energyA)
#
# wpe = compute_wpe(da)
# print(wpe)