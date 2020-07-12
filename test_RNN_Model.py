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

# data=[]
# labels=[]

# data_positive = os.listdir('Data/CallForHelp')
# data_negative = os.listdir('Data/NegativeData')
# j = 0
# # save_path = ['IMG_spectrogram/01.png',
# #                 'IMG_spectrogram/02.png',
# #                 'IMG_spectrogram/03.png',
# #                 'IMG_spectrogram/04.png',
# #                 'IMG_spectrogram/05.png']
# # save_path_negative = ['IMG_spectrogram_negative/Negative01.png',
# #                         'IMG_spectrogram_negative/Negative02.png',
# #                         'IMG_spectrogram_negative/Negative03.png',
# #                         'IMG_spectrogram_negative/Negative04.png']
# for i in data_positive :
#     s, r = librosa.load('Data/CallForHelp/'+i,sr=16000)
#     data.append(s)
    
#     labels.append(1)
    
# for i in data_negative :
#     sig, rate = librosa.load('Data/NegativeData/'+i,sr=16000)
#     data.append(sig)
#     labels.append(0)

# data.reshape(1,-1)
# print(len(data))
# print(data)

# print(len(labels))

# print(labels)
# print('---------------------------')
# #print(r)
# print(s.shape)
# print(s)
# print(len(s))

# Cells = np.asarray(data)
# labels = np.asarray(labels)
# # newCells = tf.convert_to_tensor(Cells,dtype = tf.float32)
# print(Cells)
# # print('new Cells : ',newCells)
# Cells.reshape(-1,1)
# print('Cells : ',Cells)
# #print('Cells shape : ',)
# # np.save("newCells",Cells)
# # np.save("newlabels",labels)

# # Cells=np.load('newCells.npy')
# # labels=np.load('newlabels.npy')

# a=np.arange(Cells.shape[0])
# np.random.shuffle(a)

# Cells=Cells[a]
# labels=labels[a]

# num_classes=len(np.unique(labels))
# len_data=len(Cells)

# (x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]
# print(len_data)
# print(Cells)

# # x_train = x_train.astype('float32')/255 
# # x_test = x_test.astype('float32')/255
# train_len=len(x_train)
# test_len=len(x_test)

# (y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]

# print('y train :',y_train)
# print('y test',y_test)
# print('x test',x_test)
# print(x_train.shape)
# print(y_train.shape)
# print('x train',x_train)
# print('x train shape : ',x_train.shape[1:])

def RNN_model(N_CLASSES=10, SR=16000, DT=1.0):
    # nCategories = 2
    # srate = 16000
    # iLen = 160000
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

