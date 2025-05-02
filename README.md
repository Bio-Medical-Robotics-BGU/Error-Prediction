# Error-Prediction

This repository contains the codes for error prediction in the ring tower transfer task from the long term learning experiment.
This repository uses the codes and generated labels from the following repository:
https://github.com/Bio-Medical-Robotics-BGU/Error-Detection

The code is split into several folders based on purpose:

Kinematic Segment Codes - the codes that create the kinematic segments and label them based on the labeled videos (using the vectors created for the video samples). The results are visualized and the overlap and advance parameters are set.

Create Datasets Codes - the codes that split the kinematic segments into train, validation and test sets using and 60-20-20 split. The training data can be "augmented" by increasing the overlap.

Neural Network - the code for the LSTM network.

LOUO - all the codes for testing the network with the leave one user out paradigm. This includes creating the datasets and the loop of the LSTM network.

Partial Data - the data from the three participants who participated in part of the experiment (and not all six months) was not uploaded with the original dataset. We upload them here split into three subfolders, based on if they belonged to our train, validation or test set. No additional information regarding participant code, month or time of day is provided due to anonymity considerations.
