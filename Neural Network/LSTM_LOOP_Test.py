''' This is the first try to create a network that will predict when collision errors will occur in the ring tower transfer task'''

# %% Imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import sklearn.metrics as sk

import os
import numpy as np
import scipy.io

# %% Paths and parameters

# Replace the following directory with the project directory
project = r"D:\OneDrive\lab\PhD\python\ErrorPrediction"
# project = r"C:\Users\hanna\OneDrive\lab\PhD\python\ErrorPrediction"


base = r"D:\OneDrive\MATLAB\lab\PhD\ErrorPrediction"
# base = r"C:\Users\hanna\OneDrive\MATLAB\lab\PhD\ErrorPrediction"

MatPath = os.path.join(base, "DatasetsTrainValTest_OneSplit")

seg_length = 50
max_overlap = 0
advance = 1
Tool = 'PSM'
ns = 'Standardized'

# %% Loading training and validation data
os.chdir(MatPath)
TrainKinematics = scipy.io.loadmat(f'AllTrainSignals{ns}_{Tool}_{int(max_overlap*100)}_len{seg_length}ad_{advance}_50Hz.mat')[f'AllTrainSignals{ns}']
TrainLabels = scipy.io.loadmat(f'AllTrainLabels_{int(max_overlap*100)}_len{seg_length}ad_{advance}_50Hz.mat')['AllTrainLabels']

ValKinematics = scipy.io.loadmat(f'AllValSignals{ns}_{Tool}_{int(max_overlap*100)}_len{seg_length}ad_{advance}_50Hz.mat')[f'AllValSignals{ns}']
ValLabels = scipy.io.loadmat(f'AllValLabels_{0*100}_len{seg_length}ad_{advance}_50Hz.mat')['AllValLabels']

TestKinematics = scipy.io.loadmat(f'AllTestSignals{ns}_{Tool}_{int(max_overlap*100)}_len{seg_length}ad_{advance}_50Hz.mat')[f'AllTestSignals{ns}']
TestLabels = scipy.io.loadmat(f'AllTestLabels_{0*100}_len{seg_length}ad_{advance}_50Hz.mat')['AllTestLabels']

# calculating class imbalance
class_imbalance = ((len(np.flatnonzero(TrainLabels == 0)) + len(np.flatnonzero(ValLabels == 0)))/
                      (len(np.flatnonzero(TrainLabels == 1)) + len(np.flatnonzero(ValLabels == 1))))
class_weight = {0: 1, 1: 2}


# %% checking for nans and infs
(np.isnan(TrainKinematics)).any()
(np.isnan(ValKinematics)).any()
(np.isnan(TestKinematics)).any()

(np.isinf(TrainKinematics)).any()
(np.isinf(ValKinematics)).any()
(np.isinf(TestKinematics)).any()

# %% Feature Selection
inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
TrainKinematics = np.squeeze(TrainKinematics[:, :, inds])
ValKinematics = np.squeeze(ValKinematics[:, :, inds])
TestKinematics = np.squeeze(TestKinematics[:, :, inds])

TrainKinematics = np.concatenate((TrainKinematics, ValKinematics), axis = 0)
TrainLabels = np.concatenate((TrainLabels, ValLabels), axis = 0)

# %% for informatino
print(f'Train: The number of zero segments is {len(np.flatnonzero(TrainLabels == 0))}')
print(f'Train: The number of one segments is {len(np.flatnonzero(TrainLabels == 1))}')

print(f'Test: The number of zero segments is {len(np.flatnonzero(TestLabels == 0))}')
print(f'Test: The number of one segments is {len(np.flatnonzero(TestLabels == 1))}')

# %% LSTM
lstm1 = 128
lstm2 = 64

do = 0.2
rdo = 0.0
reg = 0.001

AllAccs = np.zeros((3, 1))
AllTPRs = np.zeros((3, 1))
AllTNRs = np.zeros((3, 1))

for j in range(0, 3):
    
    
    #################################################################
    #for training on the smaller sized dataset (equal to that of A=25)
    # Tr0 = 2656
    # Tr1 = 1254

    # all1 = np.flatnonzero(TrainLabels == 1)
    # all0 = np.flatnonzero(TrainLabels == 0)
     
    # inds1 = np.random.choice(all1, Tr1, replace = False)
    # inds0 = np.random.choice(all0, Tr0, replace = False)

    # TrainKinematics1 = TrainKinematics[inds1, :, :]
    # TrainLabels1 = TrainLabels[inds1]
    
    # TrainKinematics0 = TrainKinematics[inds0, :, :]
    # TrainLabels0 = TrainLabels[inds0]

    # TrainKinematicsT = np.concatenate((TrainKinematics1, TrainKinematics0), axis = 0)
    # TrainLabelsT = np.concatenate((TrainLabels1, TrainLabels0), axis = 0)
    
    #################################################################
    K.clear_session()
    
    Input1 = Input(shape=(TrainKinematics.shape[1], TrainKinematics.shape[2]))
    x1 = Bidirectional(LSTM(lstm1, activation='tanh', return_sequences=True,
                            dropout=do, recurrent_dropout=rdo, kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))(Input1)
    
    x2 = BatchNormalization(axis=-1, momentum=0.99,
                            epsilon=0.001, center=True, scale=True)(x1)
    
    
    x3 = Bidirectional(LSTM(lstm2, activation='tanh', return_sequences=False,
                            dropout=do, recurrent_dropout=rdo, kernel_regularizer=l2(reg), bias_regularizer=l2(reg)))(x2)
    
    
    out = Dense(1, activation='sigmoid')(x3)
    
    model = Model(inputs=Input1, outputs=out)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    
    model.summary()
    
    history = model.fit(TrainKinematics, TrainLabels,
                        validation_data=(TestKinematics, TestLabels),
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
    
    
    outs = model.predict(TestKinematics)
    preds = np.zeros_like(outs)
    for i in range(outs.shape[0]):
        if outs[i] > 0.5:
            preds[i] = 1
    Cmat = sk.confusion_matrix(TestLabels, preds)
    acc = (Cmat[0, 0] + Cmat[1, 1]) / np.sum(Cmat)
    tpr = Cmat[1, 1] / np.sum(Cmat[1, :])
    tnr = Cmat[0, 0] / np.sum(Cmat[0, :])
    
    disp = sk.ConfusionMatrixDisplay(Cmat)
    disp.plot()
    plt.show()
    
    
    AllAccs[j] = acc
    AllTPRs[j] = tpr
    AllTNRs[j] = tnr
        
    os.chdir(project)
    model.save_weights(f'Weights_BiLstm_128_64_do0.2rcdo0.0_reg0.001_PSM_lr0.0001_bs128_cw2_e100_A{advance}_SmallestsSizeA25_Test_rep{j}.weights.h5')

mean_acc = np.mean(AllAccs)
std_acc = np.std(AllAccs, ddof = 1)

mean_tpr = np.mean(AllTPRs)
std_tpr = np.std(AllTPRs, ddof = 1)

mean_tnr = np.mean(AllTNRs)
std_tnr = np.std(AllTNRs, ddof = 1)

print(f'Accuracy: {round(mean_acc, 2)}, {round(std_acc, 2)}')
print(f'TPR: {round(mean_tpr, 2)}, {round(std_tpr, 2)}')
print(f'TNR: {round(mean_tnr, 2)}, {round(std_tnr, 2)}')
