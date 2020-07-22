import numpy as np
import random
import itertools
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import os
import matplotlib
import pylab
import checkdir

from scipy.io import wavfile
from librosa.core import resample, to_mono
matplotlib.use('Agg')

data_ap = []
data_n_ap = []
data_st_ap = []
data_cp_ap = []
data_cs_ap = []

#อ่านไฟล์ Audio
def load_audio_file(file_path):
    data_l = os.listdir(file_path)
    input_length = 16000
    x = 1
    for i in data_l:
        rate, data_in = wavfile.read(file_path+i)
        data_in = data_in.astype(np.float32, order='F')
        try:
            tmp = data_in.shape[1]
            data_in = to_mono(data_in.T)
        except:
            pass
        data_in = resample(data_in, rate, 16000)
        data_in = data_in.astype(np.float32)
        data_ap.append(data_in)
        x+=1
    return data_ap

# def plot_time_series(data):
#     fig = plt.figure(figsize=(14,8))
#     plt.title('Raw Wave')
#     plt.ylabel('Amplitude')
#     plt.plot(np.linspace(0, 1, len(data)), data)
#     plt.show()

# ทำ Noise Injection
def add_noise(data):
    for j in range(3):
        n_f = (j+1)*300
        wn = np.random.randn(len(data))
        data_wn = data+n_f*wn
        data_n_ap.append(data_wn)
    return data_n_ap

# ทำ Shifting time
def shifting_time(data):
    sampling_rate = 16000
    #shift_direction = 'both'
    for r_st in range(3):
        shift_max = (r_st+1)*0.375
        shift_min = r_st*0.375
        shift = np.random.randint((sampling_rate*shift_min)+1, sampling_rate*shift_max)
        # if shift_direction == 'right':
        #     shift = -shift
        # elif shift_direction =='both':
        #     direction = np.random.randint(0,2)
        #     if direction == 1:
        #         shift = -shift

        data_st = np.roll(data, shift)

        if shift > 0:
            data_st[:shift] = 0
        else:
            data_st[shift:] = 0

        data_st_ap.append(data_st)
    return data_st_ap



# ทำ Changing Pitch
def changing_pitch(data):
    sampling_rate = 16000
    for r_cp in range(3):
        pitch_factor = (r_cp+1)*0.5
        data_cp = librosa.effects.pitch_shift(data, sampling_rate,pitch_factor)
        data_cp_ap.append(data_cp)
    return  data_cp_ap

# ทำ Changing Speed
def changing_speed(data):
    for r_cs in range(3):
        speed_factor = 1+((r_cs+1)*0.15)
        data_cs = librosa.effects.time_stretch(data,speed_factor)
        data_cs_ap.append(data_cs)
    return data_cs_ap


# ฟังก์ชันที่เรียกการทำงานฟังก์ชันต่าง ๆ
def dataAug(path=None, save_path=None):
    load_audio_file(path+'/')

    for i in data_ap:
        add_noise(i)
    for i in data_n_ap:
        data_ap.append(i)
    for i in data_ap:
        shifting_time(i)
    for i in data_st_ap:
        data_ap.append(i)    
    for i in data_ap:
        changing_pitch(i)
    for i in data_cp_ap:
        data_ap.append(i)
    for i in data_ap:
        changing_speed(i)
    for i in data_cs_ap:
        data_ap.append(i)
    count=1
    for i in data_ap:
        new_i = i
        new_i = new_i.astype(np.int16)
        wavfile.write(save_path+'/'+str(count)+'.wav', 16000, new_i)
        count+=1

# ฟังก์ชันที่ใช้เช็ก path และเซตค่าต่าง ๆ เพื่อให้ฟังก์ชันอื่นง่ายต่อการเรียกใช้ เวลาเรียกใช้ DataAugmentation.py จะเรียกกับฟังก์ชันนี้โปรแกรมจะทำงานทุกขั้นตอนของ Data Augmentation
def dataAugmentation(path = None, save_path = None):
    data = os.listdir(path)
    checkdir.check_dir(save_path)
    classes = os.listdir(path)
    for i in classes:
        data_ap.clear()
        data_n_ap.clear()
        data_st_ap.clear()
        data_cp_ap.clear()
        data_cs_ap.clear()
        target_dir = os.path.join(save_path, i)
        checkdir.check_dir(target_dir)
        src_dir = os.path.join(path, i)
        dataAug(src_dir,target_dir)

if __name__ == "__main__":
    load_audio_file(file_path+'/')
    for i in data_ap:
        add_noise(i)
    for i in data_n_ap:
        data_ap.append(i)
    for i in data_ap:
        shifting_time(i)
    for i in data_st_ap:
        data_ap.append(i)        
    for i in data_ap:
        changing_pitch(i)
    for i in data_cp_ap:
        data_ap.append(i)
    for i in data_ap:
        changing_speed(i)
    for i in data_cs_ap:
        data_ap.append(i)
    for i in data_ap:
        new_i = i
        new_i = new_i.astype(np.int16)
        wavfile.write(save_path+'/'+str(count)+'.wav', 16000, new_i)
        