# import numpy as np
# from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
from scipy.io import wavfile
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from Model import RNN_model
from tqdm import tqdm
from glob import glob
import argparse

#คลาส Generator ข้อมูล และฟังก์ชันในการเทรนข้อมูล 
# Data Generator with keras 

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, wav_paths, labels, sr, dt, n_classes, batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.wav_paths)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]
        X = np.empty((self.batch_size, 1, int(self.sr*self.dt)), dtype=np.int16)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(1,-1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

#ฟังก์ชันในการเทรนข้อมูล
def train(args):
    src_root = args.src_root
    dst_root = args.dst_root
    sr = args.sample_rate
    dt = args.delta_time
    batch_size = args.batch_size
    model_type = args.model_type
    params = {'N_CLASSES':len(os.listdir(args.src_root)),
                'SR':sr,
                'DT':dt}
    models = {'RNN_model':RNN_model(**params)}
    assert model_type in models.keys(), '{} not an available model'.format(model_type)
    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
    classes = sorted(os.listdir(args.src_root))
    le = LabelEncoder()
    le.fit(classes)
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    labels = le.transform(labels)

    wav_train, wav_val, label_train, label_val = train_test_split(wav_paths, labels, test_size=0.1, random_state=0)

    tg = DataGenerator(wav_train, label_train, sr, dt, len(set(label_train)),batch_size=batch_size)
    
    vg = DataGenerator(wav_val, label_val, sr, dt, len(set(label_val)), batch_size=batch_size)
    
    model = models[model_type]
    cp = ModelCheckpoint('model/{}.h5'.format(model_type), monitor='val_loss',
                            save_best_only = True, save_weights_only = False,
                            mode = 'auto', save_freq='epoch', verbose=1)
    
    model.fit(tg, validation_data=vg, epochs = 2, verbose=1) # เพิ่ม-ลด Epochs โดยการเซ็ตค่าที่ epochs
    model.save(dst_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Call for help Classification Training')
    parser.add_argument('--model_type',type=str, default='RNN_model',help='model to run lstm')
    parser.add_argument('--src_root',type=str,default='Audio/cleanHelp')
    parser.add_argument('--dst_root', type=str, default='Model/RNN_Help.h5',help='Model path')
    parser.add_argument('--batch_size',type=int, default=1,help='batch_size')
    parser.add_argument('--delta_time','-dt',type=float,default=2.0,help='time in seconds to sample audio')
    parser.add_argument('--sample_rate','-sr',type=int,default=16000, help='sample rate')
    args, _ = parser.parse_known_args()
    train(args)