"""
Code for running and training model
"""

import numpy as np

from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
import tensorflow as tf

from model import *
from load_data import load_data
import config

#### UNUSED AS OF NOW BECAUSE TOO MUCH MEMORY CONSUMPTION
def preprocess_data(X_train, y_train, X_val, y_val):
    # data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=config.rotation,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # actual transformation step
    # datagen.fit(X)

    print("# Generating data flow.")

    # input into model fit
    return (datagen.flow(X_train, y_train, config.batch_size),
        datagen.flow(X_val, y_val, config.batch_size))

def train(X_train, y_train, X_val, y_val):
    """
    Trains our CNN
    """
    # create base model
    model = resnet_model()

    # optimizers: adam, rmsprop, sgd, etc.
    model.compile(optimizer=config.optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy','top_k_categorical_accuracy'])

    # callbacks for training
    cb_early_stop = EarlyStopping(monitor="val_loss", patience=config.patience)
    cb_checkpoint = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5")
    cb_csv = CSVLogger("training.log")

    # preprocessed data niceness
    inputs_train, inputs_val = preprocess_data(X_train, y_train, X_val, y_val)

    print("# Data loaded. Beginning training.")

    history = model.fit_generator(
        inputs_train, epochs=config.epochs,
        steps_per_epoch=X_train.shape[0] / config.batch_size,
        validation_steps=X_val.shape[0] / config.batch_size,
        verbose=1, validation_data=inputs_val,
        callbacks=[cb_early_stop, cb_checkpoint, cb_csv])

    return model, history

def evaluate(X, y, model):
    return model.evaluate(X, y)

def predict(X, model):
    results = model.predict(X)
    return results

