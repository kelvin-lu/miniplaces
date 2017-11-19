"""
Code for loading in data
"""

import os
import numpy as np
import scipy.misc
import h5py

from keras.utils import to_categorical

from DataLoader import *
import config

# Dataset Parameters
load_size = config.size
fine_size = config.size-31
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

classes = 100

# Construct dataloader
opt_data_train = {
    'data_h5': config.h5_train,
    'data_root': config.data_root,
    'data_list': config.data_train_list,
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_h5': config.h5_val,
    'data_root': config.data_root,
    'data_list': config.data_val_list,
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }
opt_data_test= {
    'data_h5': config.h5_test,
    'data_root': config.data_root,
    'data_list': config.data_val_list,
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

def load_data():
    loader_train = DataLoaderH5(**opt_data_train)
    loader_val = DataLoaderH5(**opt_data_val)
    loader_test = DataLoaderH5Test(**opt_data_test)

    X_train, y_train = loader_train.get_data()
    X_val, y_val = loader_val.get_data()
    X_test = loader_test.get_data()

    # normalize X
    X_train = X_train.astype(np.float32) / 255.
    X_val = X_val.astype(np.float32) / 255.
    X_test = X_test.astype(np.float32) / 255.

    # change y to one-hot
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    return X_train, y_train, X_val, y_val, X_test

