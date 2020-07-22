from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import optimizers
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.regularizers import l2

import os 
import matplotlib
matplotlib.use('Agg')
import pylab
import librosa
import librosa.display
import numpy as np

# ฟังก์ชันสร้างโมเดล RNN โดยมี input คือ จำนวน class ที่ต้องการจำแนก sampling rate ของไฟล์ audio และจำนวนเวลาของไฟล์ audio 
def RNN_model(N_CLASSES=2, SR=16000, DT=2.0):

    rnn_func = L.LSTM

    inputs = L.Input(shape=(1,int(SR*DT)),name='input')

    x = L.Reshape((1,-1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128,
                        padding='same', sr=SR,n_mels=80,
                        fmin=40,fmax=SR/2,power_melgram=1.0,
                        return_decibel_melgram=True, trainable_fb=False,
                        trainable_kernel=False,
                        name='mel_stft')

    m.trainable = False

    x = m(x)

    x = Normalization2D(int_axis=0,name='mel_stft_norm')(x)

    x = L.Permute((2,1,3))(x)

    x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    x = L.Lambda(lambda q: K.squeeze(q,-1),name='squeeze_last_dim')(x)

    x = L.Bidirectional(rnn_func(64, return_sequences=True))(x)

    x = L.Bidirectional(rnn_func(64, return_sequences=True))(x)

    xFirst = L.Lambda(lambda q: q[:,-1])(x)
    query = L.Dense(128)(xFirst)

    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)

    attVector = L.Dot(axes=[1, 1])([attScores, x])

    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(32)(x)

    output = L.Dense(N_CLASSES, activation='softmax',name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    model.summary()



    # model.fit(x_train,y_train,batch_size=32,epochs=5
    #             #callbacks=[earlystopper, checkpointer, lrate]
    #             )
    # from keras.models import load_model
    # model.save('test_model_RNN.h5')

    ##### OLD Version RNN Model  ##### 

    # i = L.Input(shape=(1,int(SR*DT)), name='input')
    # x = Melspectrogram(n_dft=512, n_hop=160, padding='same', sr=SR, n_mels=128, fmin=0.0, 
    #                     fmax=SR/2, power_melgram=1.0, return_decibel_melgram=True,
    #                     trainable_fb=False, trainable_kernel=False,name='melbands')(i)
    # x = Normalization2D(str_axis='batch', name='batch_norm')(x)
    # x = L.Permute((2,1,3), name='permute')(x)
    # x = TimeDistributed(L.Reshape((-1,)), name='reshape')(x)
    # s = TimeDistributed(L.Dense(64, activation='tanh'), name='td_dense_tanh')(x)
    # x = L.Bidirectional(L.LSTM(32, return_sequences=True), name='bidirectional_lstm')(s)
    # x = L.concatenate([s, x], axis=2, name='skip_connection')
    # x = L.Dense(64, activation='relu', name='dense_1_relu')(x)
    # x = L.MaxPooling1D(name='max_pool_1d')(x)
    # x = L.Dense(32, activation='relu', name='dense_2_relu')(x)
    # x = L.Flatten(name='flatten')(x)
    # x = L.Dropout(rate=0.2, name='dropout')(x)
    # x = L.Dense(32, activation='relu', activity_regularizer=l2(0.001),name='dense_3_relu')(x)
    # o = L.Dense(N_CLASSES, activation='softmax', name='softmax')(x)

    # model = Model(inputs=i, outputs=o, name='long_short_term_memory')
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

