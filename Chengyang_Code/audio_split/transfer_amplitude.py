from pyAudioAnalysis import audioBasicIO as aIO
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as nf
import librosa
from collections import Counter


def amplitude_normalize(s):
    data = np.copy(s)
    mea = np.mean(data)
    std = np.std(data)
    da = (data-mea)/std

    return da


def reduce_noise(sigs, sr):
    freqs = nf.fftfreq(sigs.size, 1 / sr)
    complex_arry = nf.fft(sigs)
    pows = np.abs(complex_arry)
    fun_freq = freqs[pows.argmax()]  # 获取频率域中能量最高的
    noised_idx = np.where(freqs <= fun_freq/2)[0]  # 获取所有噪声的下标
    ca = complex_arry[:]
    ca[noised_idx] = ca[noised_idx]/3 # 高通滤波

    filter_sigs = nf.ifft(ca)
    filter_sigs = (filter_sigs * (2 ** 15)).astype('i2')
    return filter_sigs