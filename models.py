# -*- coding: utf-8 -*-
"""
A module for defining neural network-based Keras models as classes.
"""

# basic imports
import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re
import math
import sys
import os

# set env backend to theano
print('Importing keras and sklearn ...')
os.environ['KERAS_BACKEND']='theano'

# Keras imports
from keras import backend as K
from keras.layers import Activation
from keras.layers import Dense, Input, Flatten, BatchNormalization
from keras.layers import Embedding, Merge, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adagrad, Adam
from keras.layers import Layer
from keras.callbacks import EarlyStopping

class DeepAveragingNet(object):
    """
    A class to define static deep averaging network (DAN),
    also known as Neural Bag-of-Words (NBoW). This class defines a static model
    regarding the word embeddings. We assume context averaged vectors are
    already computed.

    # Arguments
        layers: size of each hidden layer, list of int (e.g. [400, 400])
        input_shape: a tuple for input shaper, e.g. input_shape=(400,)
        batch_normalised: a boolean to batch normalise or not
        dropout: dropout rate
    """

    def __init__(self, layers, input_dim, batch_normalised=True, dropout=2.0):
        self.layers = layers
        self.input_dim = input_dim
        self.batch_normalised = batch_normalised
        self.dropout = dropout

        self.model = self._build_model()

    def _build_model(self):
        """Build a Keras model and return a keras Sequential object."""
        print('Building a static deep averaging model (DAN) model ...')

        # create model
        model = Sequential()

        # create input and hidden layers
        for i in range(len(self.layers) - 1):
            if i == 0:
                # input layer
                model.add(Dense(self.layers[i], input_shape=(self.input_dim,)))
            else:
                # hidden layers
                model.add(Dense(self.layers[i]))

                model.add(Dropout(self.dropout))
                model.add(Activation('relu'))

            if self.batch_normalised:
                model.add(BatchNormalization())

        # output layer
        model.add(Dense(self.layers[-1]))
        if self.batch_normalised:
	           model.add(BatchNormalization())

        model.add(Dropout(self.dropout))
        model.add(Activation('sigmoid'))

        self.n_params = model.count_params()
        print('Number of parameters in the model: ', self.n_params)

        return model
