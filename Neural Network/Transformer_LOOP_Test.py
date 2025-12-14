''' This is the first try to create a network that will predict when collision errors will occur in the ring tower transfer task'''

# %% Imports
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import sklearn.metrics as sk
from sklearn.metrics import roc_auc_score


import optuna
from optuna.integration import TFKerasPruningCallback



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

# %% Model Functions
def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims//2)) / np.float32(d_model))
    angle_rads = positions * angle_rates

    # apply sin to even indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, dtype=tf.float32)


def transformer_encoder(inputs, head_size, num_heads, d_model, ff_dim, dropout=0):
    # Attention and Normalization
    norm = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(norm, norm)
    attn = layers.Dropout(dropout)(attn)
    attn = attn + inputs
    
    
    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(attn)
    ff = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Conv1D(filters=d_model, kernel_size=1)(ff)
    x = ff + attn
    return x



def build_model(input_shape, head_size, num_heads, d_model, ff_dim, num_transformer_blocks, mlp_units, n_classes, dropout=0, mlp_dropout=0):
   
    seq_len = input_shape[0]
    feature_dim = 64

    inputs = tf.keras.Input(shape=input_shape)

    pos_encoding = positional_encoding(seq_len, feature_dim)   # shape (seq_len, feature_dim)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)        # shape (1, seq_len, feature_dim)

    embeddings = layers.Dense(feature_dim)(inputs)

    x = embeddings + pos_encoding

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, d_model, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    
    for dim in mlp_units:
        x = layers.Dense(dim, kernel_regularizer=l2(0.02), activation = "relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    outputs = layers.Dense(n_classes, activation="sigmoid")(x)
    
    return tf.keras.Model(inputs, outputs)

#creating and training models
input_shape = TrainKinematics.shape[1:]

# %% Training

AllAccs = np.zeros((5, 1))
AllTPRs = np.zeros((5, 1))
AllTNRs = np.zeros((5, 1))
AllAUCs = np.zeros((5, 1))

for j in range(0, 5):
    
    

    K.clear_session()
    
    model = build_model(
        input_shape,
        head_size=16,
        num_heads=4,
        d_model = 64, 
        ff_dim=36,
        num_transformer_blocks=5,
        mlp_units=[64, 32],
        n_classes = 1,
        mlp_dropout=0.1,
        dropout=0.3)
    
    

    from tensorflow.keras.callbacks import ReduceLROnPlateau

    lr_scheduler = ReduceLROnPlateau(
         monitor='val_loss',   # metric to watch
         factor=0.5,           # reduce LR to LR * factor
         patience=3,           # epochs with no improvement before reducing
         min_lr=1e-6,          # do not reduce below this
         verbose=1             # print updates
     )
    callbacks = [lr_scheduler]
     
    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001) 
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['acc'])

    model.summary()

    history = model.fit(TrainKinematics, TrainLabels, 
                        validation_data=(TestKinematics, TestLabels), 
                        batch_size = 128, callbacks = callbacks,
                        shuffle = True, epochs = 100, verbose = 1, class_weight = class_weight)



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
    AllAUCs[j] = roc_auc_score(TestLabels, outs)

        
 
mean_acc = np.mean(AllAccs)
std_acc = np.std(AllAccs, ddof = 1)

mean_tpr = np.mean(AllTPRs)
std_tpr = np.std(AllTPRs, ddof = 1)

mean_tnr = np.mean(AllTNRs)
std_tnr = np.std(AllTNRs, ddof = 1)

mean_auc = np.mean(AllAUCs)
std_auc = np.std(AllAUCs, ddof = 1)

print(f'Accuracy: {round(mean_acc, 2)}, {round(std_acc, 3)}')
print(f'TPR: {round(mean_tpr, 2)}, {round(std_tpr, 3)}')
print(f'TNR: {round(mean_tnr, 2)}, {round(std_tnr, 3)}')
print(f'AUC: {round(mean_auc, 2)}, {round(std_auc, 3)}')
