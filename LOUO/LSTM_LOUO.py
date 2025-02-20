# %% Imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Subtract, Bidirectional, MultiHeadAttention, LayerNormalization, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import sklearn.metrics as sk

from scipy import stats

import os
import numpy as np
import scipy.io
import pandas as pd
import random

# %% Paths and parameters

# Replace the following directory with the project directory
project = r"D:\OneDrive\lab\PhD\python\ErrorPrediction"

base = r"D:\OneDrive\MATLAB\lab\PhD\ErrorPrediction"

# project = r"C:\Users\hanna\OneDrive\lab\PhD\python\ErrorPrediction"

# base = r"C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction"
MatPath = os.path.join(base, "Datasets_LOUO")

seg_length = 50
max_overlap = 0
advance = 1
Tool = 'PSM'
ns = 'Standardized'

AllParticipants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']


# %% 
AllValMetrics = np.zeros((len(AllParticipants), 3))

# %% LSTM

def LSTM_Model(input_shape, lstm1, lstm2, do, rdo, reg):
    Input1 = Input(shape = input_shape)
    x1 = Bidirectional(LSTM(lstm1, activation='tanh', return_sequences=True,
                            dropout=do, recurrent_dropout=rdo, kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))(Input1)

    x2 = BatchNormalization(axis=-1, momentum=0.99,
                            epsilon=0.001, center=True, scale=True)(x1)


    x3 = Bidirectional(LSTM(lstm2, activation='tanh', return_sequences=False,
                            dropout=do, recurrent_dropout=rdo, kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))(x2)


    out = Dense(1, activation='sigmoid')(x3)

    model = Model(inputs=Input1, outputs=out)
    
    return model


lstm1 = 128
lstm2 = 64

do = 0.2
rdo = 0.0
reg = 0.001


class_weight = {0: 1, 1: 2}

# %% Running over all participants

for pp, p in enumerate(AllParticipants):
    print(pp, p)
    
    os.chdir(MatPath)
    
    TrainKinematics = scipy.io.loadmat(f'AllTrainSignals{ns}_LOUO_{p}_{Tool}_{int(max_overlap*100)}_len{seg_length}ad_{advance}_50Hz.mat')[f'AllTrainSignals{ns}']
    TrainLabels = scipy.io.loadmat(f'AllTrainLabels_LOUO_{p}_{int(max_overlap*100)}_len{seg_length}ad_{advance}_50Hz.mat')['AllTrainLabels']

    ValKinematics = scipy.io.loadmat(f'AllTestSignals{ns}_LOUO_{p}_{Tool}_{int(max_overlap*100)}_len{seg_length}ad_{advance}_50Hz.mat')[f'AllTestSignals{ns}']
    ValLabels = scipy.io.loadmat(f'AllTestLabels_LOUO_{p}_{max_overlap*100}_len{seg_length}ad_{advance}_50Hz.mat')['AllTestLabels']
    
    input_shape = (TrainKinematics.shape[1], TrainKinematics.shape[2])

    K.clear_session()

    model = LSTM_Model(input_shape, lstm1, lstm2, do, rdo, reg)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

    model.summary()

    history = model.fit(TrainKinematics, TrainLabels,
                        validation_data=(ValKinematics, ValLabels),
                        batch_size=128, 
                        shuffle=True, epochs=100, verbose=1, class_weight=class_weight)
    
    history = history.history

    acc = history['acc']
    val_acc = history['val_acc']

    loss = history['loss']
    val_loss = history['val_loss']

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
    
    outs = model.predict(ValKinematics)


    preds = np.zeros_like(outs)
    for i in range(outs.shape[0]):
        if outs[i] > 0.5:
            preds[i] = 1
    Cmat = sk.confusion_matrix(ValLabels, preds)
    acc = (Cmat[0, 0] + Cmat[1, 1]) / np.sum(Cmat)
    tpr = Cmat[1, 1] / np.sum(Cmat[1, :])
    tnr = Cmat[0, 0] / np.sum(Cmat[0, :])

    disp = sk.ConfusionMatrixDisplay(Cmat)
    disp.plot()
    plt.show()
    
    
    AllValMetrics[pp, 0] = acc
    AllValMetrics[pp, 1] = tpr
    AllValMetrics[pp, 2] = tnr



MeanAcc = np.mean(AllValMetrics[:, 0])*100
StdAcc = np.std(AllValMetrics[:, 0])*100

MeanTPR = np.mean(AllValMetrics[:, 1])*100
StdTPR = np.std(AllValMetrics[:, 1])*100

MeanTNR = np.mean(AllValMetrics[:, 2])*100
StdTNR = np.std(AllValMetrics[:, 2])*100