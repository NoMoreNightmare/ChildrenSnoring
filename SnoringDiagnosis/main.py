from pyAudioAnalysis import ShortTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO
import numpy as np
import plotly.graph_objs as go
import plotly
from pydub import AudioSegment
import os

import matplotlib.pyplot as plt

def MP32WAV(mp3_path, wav_path):
    """
    这是MP3文件转化成WAV文件的函数
    :param mp3_path: MP3文件的地址
    :param wav_path: WAV文件的地址
    """
    # MP3_File = AudioSegment.from_mp3(file=mp3_path).set_frame_rate(24000)
    MP3_File = AudioSegment.from_mp3(file=mp3_path)
    MP3_File.export(wav_path, format="wav")

def M4A2WAV(m4a_path, wav_path):
    M4A_File = AudioSegment.from_file(m4a_path, 'm4a')
    M4A_File.export(wav_path, format="wav")

def MP42WAV(mp4_path, wav_path):
    MP4_File = AudioSegment.from_file(mp4_path, 'mp4')
    MP4_File.export(wav_path, format="wav")

def AAC2WAV(aac_path, wav_path):
    AAC_File = AudioSegment.from_file(aac_path, 'aac')
    AAC_File.export(wav_path, format="wav")

def File2WAV(file_path, wav_path):
    ext = file_path.split('.')[-1]
    if ext == 'mp3':
        MP32WAV(file_path, wav_path)
    elif ext == 'm4a':
        M4A2WAV(file_path, wav_path)
    elif ext == 'mp4':
        MP42WAV(file_path, wav_path)
    elif ext == 'aac':
        AAC2WAV(file_path, wav_path)
    else:
        print(file_path, ext)

def audio_features(file_path, prefix):
    out_folder = 'test'
    print(file_path[len(prefix):])

    wav_path = out_folder+'/'+file_path[len(prefix):]+'.wav'
    File2WAV(file_path, wav_path)

    # read audio data from file
    # (returns sampling freq and signal as a numpy array)
    [fs, s] = aIO.read_audio_file(wav_path)
    s = aIO.stereo_to_mono(s)
    # play the initial and the generated files in notebook:
    # IPython.display.display(IPython.display.Audio("data/object.wav"))
    # print duration in seconds:
    # print(len(s) , float(fs))
    duration = len(s) / float(fs)
    print(f'duration = {duration} seconds')
    # extract short-term features using a 50msec non-overlapping windows
    win, step = 0.050, 0.050
    # print(s, fs, int(fs * win), int(fs * step))
    [f, fn] = aF.feature_extraction(s, fs, int(fs * win),
                                    int(fs * step))
    # print(f'{f.shape[1]} frames, {f.shape[0]} short-term features')
    # print('Feature names:')
    # for i, nam in enumerate(fn):
    #     print(f'{i}:{nam}')
    # plot short-term energy
    # create time axis in seconds
    # time = np.arange(0, duration - step, win)
    # get the feature whose name is 'energy'
    # energy = f[fn.index('energy'), :]
    # mylayout = go.Layout(yaxis=dict(title="frame energy value"),
    #                      xaxis=dict(title="time (sec)"))
    # plotly.offline.iplot(go.Figure(data=[go.Scatter(x=time,
    #                                                 y=energy)],
    #                                layout=mylayout))

    # plt.subplot(2,1,1); plt.plot(f[0,:]);plt.xlabel('Frame no');plt.ylabel('ZCR')
    # plt.savefig(wav_path+'.zcr.jpg')
    # # plt.show()
    # plt.close()
    for i, name in enumerate(fn):
        plt.subplot(2,1,1); plt.plot(f[i,:]);plt.xlabel('Frame no');plt.ylabel(name)
        plt.savefig(wav_path+'.'+name+'.jpg')
        # plt.show()
        plt.close()





def main():
    # loop over all sound tracks, convert videos/sounds to wav files
    root = 'recordings/'
    all_folders = [x[0] for x in os.walk(root)]

    all_files = [os.listdir(dir_path) for dir_path in all_folders]
    # print(all_files)
    for dir_path in all_folders:
        for file in os.listdir(dir_path):
            # print(dir_path+'/'+file)
            file_path = dir_path+'/'+file
            file_type_break = file_path.split('.')
            file_type = file_type_break[1] if len(file_type_break) == 2 else ''
            if file_type in ['mp3','m4a','mp4','aac']:
                audio_features(file_path, root)

    # audio_features('recordings/腺样体伴扁桃体肥大/20.mp3', root)
    # audio_features('recordings/腺样体伴扁桃体肥大/1.m4a', root)
    # audio_features('recordings/腺样体伴扁桃体肥大/2（1）.mp4', root)
    # audio_features('recordings/腺样体伴扁桃体肥大/21.aac', root)


if __name__ == '__main__':
    main()
