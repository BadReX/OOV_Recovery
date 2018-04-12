# -*- coding: utf-8 -*-
"""
A program to perform experiments.
"""

import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict, Counter
import re
import math
import sys
import h5py

import preprocessing
import models

from keras.optimizers import Adagrad, Adam
from keras.layers import Layer
from keras.callbacks import EarlyStopping

# other imports
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


#### read wikipedia documents
print('Reading data from disk ...')
DATA_PATH = '/home/babdulla/DNN/wiki-data/wikidocs.OOVs.shuffled.csv'
wikidocs = pd.read_csv(DATA_PATH, sep=',')
print('Dimensions of data frame (r, c):', wikidocs.shape)

#### get word2cluster
# read clusters
print('Reading word clusters from disk ...')
CLUSTER_PATH = '/home/babdulla/DNN/wiki-processed/wiki-clusters.1k'
with open(CLUSTER_PATH) as txtFile:
    lines = txtFile.readlines()

word2cluster = defaultdict(lambda: -1)

for line in lines:
    word, cluster = line.split()
    word2cluster[word] = cluster

#### make dataset from wikipedia documents
print('Prepare data lists ...')
docs = []
labels = []
relevant_OOVs= []

for idx in range(wikidocs.words.shape[0]): # 50000
    doc_str = wikidocs.words[idx]

    # this condition was added because 2 rows were NaN for unknown reason!
    if isinstance(doc_str, str):
        docs.append(doc_str)
        labels.append(wikidocs.clusters[idx])
        relevant_OOVs.append(wikidocs.OOVs[idx])

#### obtain and analyse OOVs
OOVs = [OOV for OOV_list in  wikidocs.OOVs for OOV in OOV_list.split()]
COUNT_OOV = Counter(OOVs)

i = 506
print(wikidocs[i:i+10])

#### remove instances where only the default class exist
labels_without_default = [[int(i) for i in l.split() if i != '-1']  for l in labels]

# added this to eliminate the docs with only the default class

docsX = []
labelsY = []
relevant_OOVs_Y = []

for idx in range(len(labels_without_default)):
    if labels_without_default[idx]:
        docsX.append(docs[idx])
        labelsY.append(labels_without_default[idx])
        relevant_OOVs_Y.append(relevant_OOVs[idx])

#### test case
assert len(docsX) == len(labelsY), 'Doc - Label length mismatch.'

print('Test OK!\n')

#### obtain vocab from wikiedia documents
print('Counting the number of unique words in the data ...')
vocab = set([w for d in docsX for w in d.split()])
print('Number of unique words (vocab) is ',  len(vocab))


#### get word2index dictionary
# read word index
print('Reading word indexing from disk ...')
W2I_PATH = '/users/babdulla/oov_recovery/metadata/vocab.word2idx'
with open(W2I_PATH) as txtFile:
    lines = txtFile.readlines()

word2index = {}

for line in lines:
    word, index = line.split()
    word2index[word] = int(index)


#### tokenization and sequencing
print('Tokenizing text in the data and make sequences ...')
fast_tokenizer = preprocessing.Fast_Tokenizer(word2index)

doc_seqs = fast_tokenizer.documents_to_sequences(docsX)

print('___________', len(doc_seqs), type(doc_seqs), len(docsX))

#### test case
print('Check if the word_index of tokenizer object is same as vocab ...')
assert len(fast_tokenizer.word_index) == len(vocab), "word index length mismatch."
assert max(fast_tokenizer.word_index.values()) == len(vocab), "max word index != vocab size"
print(max(fast_tokenizer.word_index.values()), len(vocab))
print('Found {0} unique tokens.'.format(len(word2index)))
print('Vocab tests passed! No problems.')

#### Labels Binarization
print('Binarizing and transforming the labels with MultiLabelBinarizer ...')
# vectorize labels in the output
mlb = MultiLabelBinarizer(classes=range(1000))
bin_labels = mlb.fit_transform(labelsY)


#### test case
assert bin_labels.shape[0] == len(docsX), "input/output data mismatch."


#### WORD EMBEDDINGS
# read pre-trained word embeddings
print('Read word embeddings from disk ...')
w2v_DIR = "/home/babdulla/DNN/wiki-processed/word-vectors.400d"

embeddings_index = {}

with open(w2v_DIR, "r") as eFile:
    for line in eFile:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Total %s word vectors are in the embedding matrix.' % len(embeddings_index))

# make embedding matrix
embedding_matrix = np.zeros((len(fast_tokenizer.word_index) + 1, 400))

print('Initializing emnbedding matrix ...')
for word, i in fast_tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Shape of the embedding tensor:', embedding_matrix.shape)

#### MAKE GLOBAL AVERAGED VECTORS FOR DOCUMENTS
# make averaged context vectors (global average pooling)
print('Generating context vectors ...')
context_vectors = np.zeros((len(doc_seqs), 400))

for i in range(len(doc_seqs)):
    seq = doc_seqs[i]
    context_vectors[i] = np.sum([embedding_matrix[w] for w in seq],  axis=0)
    context_vectors[i] /= len(seq)

print('Shape of the contexts tensor:', context_vectors.shape)

assert len(context_vectors) == len(doc_seqs), "Length mismatchdue to context vectors!"


#### TRAINING NEURAL MODEL
# make train, dev, and test splits
print('Making train, dev, and test splits ...')
DEV_SPLIT  = 0.01
TEST_SPLIT = 0.01

ndev_samples  = int(DEV_SPLIT * context_vectors.shape[0])
ntest_samples = int(TEST_SPLIT * context_vectors.shape[0])

dev_idx = -ndev_samples - ntest_samples

x_train = context_vectors[:dev_idx]
y_train = bin_labels[:dev_idx]

x_dev = context_vectors[dev_idx:-ndev_samples]
y_dev = bin_labels[dev_idx:-ndev_samples]

x_test = context_vectors[-ndev_samples:]
y_test = bin_labels[-ndev_samples:]

y_test_OOVs = [set(OOVs.split()) for OOVs in relevant_OOVs_Y[-ndev_samples:]]

# some stats about the splits
print('Train split:', x_train.shape)
print('Dev split:', x_dev.shape)
print('Test split:', x_test.shape)

#### BUILD NEURAL MODEL

print('Building a neural model ...')
input_dim = 400
dropout_rate = 0.2
layers = [1000, 1000]
batch_normalised = True

neural_net = models.DeepAveragingNet(layers, input_dim, batch_normalised, dropout_rate)


print('Model summary.:')
neural_net.model.summary()

adam = Adam()
neural_net.model.compile(loss='binary_crossentropy',
    optimizer=adam,
    metrics=['accuracy'])

print('Training a neural model ...')
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop]

model_info = neural_net.model.fit(x_train,
    y_train,
    batch_size=124,
    callbacks=callbacks_list,
    epochs=20,
    validation_data=(x_dev, y_dev),
    verbose=1)

print('Evaluating the model ...')
# use the model to make preditions
print("Using the neural model to make predictions ...")
nn_preds = neural_net.model.predict(x_test)

idx_set = [6000, 4127, 49, 59, 5008, 777, 7484, 11155]

fig, axs = plt.subplots(len(idx_set),1,figsize=(8,14))

for i in range(len(idx_set)):
    axs[i].plot(y_test[idx_set[i]], color='c', alpha=1, linewidth=1)
    axs[i].plot(nn_preds[idx_set[i]], color='r', alpha=1, linewidth=1)
    axs[i].axhline(y=0.5, color='k', linestyle='--', linewidth=0.75, alpha=1)
    axs[i].axhline(y=0.3, color='green', linestyle='--', linewidth=0.75, alpha=1)
    axs[i].set_yticks([0, 0.3, 0.5, 1])
    axs[i].set_xticks([])

plt.show()
