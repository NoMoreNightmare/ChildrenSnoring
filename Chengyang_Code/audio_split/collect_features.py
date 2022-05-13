import os
from pyAudioAnalysis import audioBasicIO as aIO
import transfer_amplitude as ta
import numpy as np
from pyAudioAnalysis import ShortTermFeatures as aF
import matplotlib.pyplot as plt
import librosa
from scipy import stats
# import opensmile
from wavelet_compute import *
from scipy import signal
from skimage import feature


def read_wav_files(root):
    fs_list = []
    signals = []
    #specflat_list = []
    rmse_list = []
    #zcr_list = []
    #speccent_list = []
    #f0_list = []
    #rolloff_list = []
    #onset_list = []
    #mfcc_list = []
    #f0_list = []
    print("hhh")
    all_folders = [x[0] for x in os.walk(root)]
    all_files = [os.listdir(dir_path) for dir_path in all_folders]
    print(all_files)
    for dir_path in all_folders:
        for file in os.listdir(dir_path):
            print(dir_path+'/'+file)
            file_path = dir_path+'/'+file
            file_type_break = file_path.split('.')
            file_type = file_type_break[len(file_type_break)-1]
            if file_type in ['wav']:
                [fs, s] = aIO.read_audio_file(file_path)
                s = aIO.stereo_to_mono(s)

                da = ta.reduce_noise(s, fs)
                da = ta.amplitude_normalize(da)

                frequencies, times, spectrogram = signal.spectrogram(da, fs)
                spectrogram = np.log(spectrogram)
                lbp = feature.local_binary_pattern(spectrogram, P=6, R=2)
                np.save(file_type_break[0]+'_lbp.npy', lbp)

                #wte, wteA = compute_wte(da)
                #wpe = compute_wpe(da)
                #np.save(file_type_break[0]+'_wte.npy', wte)
                #np.save(file_type_break[0]+'_wteA.npy', wteA)
                #np.save(file_type_break[0]+'_wpe.npy', wpe)

                #ans = smile.process_signal(
                #    da,
                #    fs
                #)
                #ans.to_csv(file_type_break[0]+'.csv', encoding='utf-8')
                #mfcc = librosa.feature.mfcc(y=da)
                #mfcc_list.append(np.mean(mfcc))
                #ons = librosa.onset.onset_strength(y=da)
                #onset_list.append(np.mean(ons))
                #rf = librosa.feature.spectral_rolloff(y=da)
                #rolloff_list.append(np.mean(rf[0]))
                # f0, vf, vp = librosa.pyin(y=da, fmax=librosa.note_to_hz('C7'), fmin=librosa.note_to_hz('C2'))
                #print(f0)
                # f0[np.isnan(f0)] = 0
                # f0_list.append(np.mean(f0))
                #speccent = np.mean(librosa.feature.spectral_centroid(y=da))
                #speccent_list.append(speccent)
                #zcr = np.mean(librosa.feature.zero_crossing_rate(y=da))
                #zcr_list.append(zcr)
                o_rmse = librosa.feature.rms(y=da)
                rmse = np.mean(o_rmse)
                rmse_list.append(rmse)
                #sf = librosa.feature.spectral_flatness(y=da)
                #specflat_list.append(sum(sf[0]))
                signals.append(da)
                fs_list.append(fs)
    # rmse_list = f0_list
    return fs_list, signals, rmse_list


def extract_wav_feature(file):
    file_path = file
    rmse_list=[]
    signals=[]
    fs_list=[]
    file_type_break = file_path.split('.')
    file_type = file_type_break[len(file_type_break)-1]
    if file_type in ['wav']:
        [fs, s] = aIO.read_audio_file(file_path)
        s = aIO.stereo_to_mono(s)

        da = ta.reduce_noise(s, fs)
        da = ta.amplitude_normalize(da)

        frequencies, times, spectrogram = signal.spectrogram(da, fs)
        spectrogram = np.log(spectrogram)
        lbp = feature.local_binary_pattern(spectrogram, P=6, R=2)
        np.save(file_type_break[0] + '_lbp.npy', lbp)

        o_rmse = librosa.feature.rms(y=da)
        rmse = np.mean(o_rmse)


        rmse_list.append(rmse)

        # sf = librosa.feature.spectral_flatness(y=da)
        # specflat_list.append(sum(sf[0]))
        signals.append(da)
        fs_list.append(fs)

    return fs_list, signals, rmse_list

def extract_wav_file_with_name(file):
    def extract_wav_feature(file):
        file_path = file
        rmse_list = []
        signals = []
        fs_list = []
        file_type_break = file_path.split('.')
        file_type = file_type_break[len(file_type_break) - 1]
        if file_type in ['wav']:
            [fs, s] = aIO.read_audio_file(file_path)
            s = aIO.stereo_to_mono(s)

            da = ta.reduce_noise(s, fs)
            da = ta.amplitude_normalize(da)

            frequencies, times, spectrogram = signal.spectrogram(da, fs)
            spectrogram = np.log(spectrogram)
            lbp = feature.local_binary_pattern(spectrogram, P=6, R=2)
            np.save(file_type_break[0] + '_lbp.npy', lbp)

            o_rmse = librosa.feature.rms(y=da)
            rmse = np.mean(o_rmse)

            temp = [file_path, rmse]
            rmse_list.append(temp)

            # rmse_list.append(rmse)

            # sf = librosa.feature.spectral_flatness(y=da)
            # specflat_list.append(sum(sf[0]))
            signals.append(da)
            fs_list.append(fs)

        return fs_list, signals, rmse_list