# -*- coding: utf-8 -*-
"""
A module for text preprocessing.
Code based on  keras/keras/preprocessing/text.py
"""

import string
import sys
import warnings
from collections import OrderedDict
from hashlib import md5

class Fast_Tokenizer(object):
    """
    Text tokenization utility class.

    Given a mapping dictionary from word (string) to identity (integer),
    this class vectorizes a text corpus, by turning each document into
    a sequence of integers

    # Arguments
    word2index: a mapping dict from word to index.
    """

    def __init__(self, word2index, char_level=False, **kwargs):
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        self.word_index = word2index
        self.char_level = char_level
        self.index_docs = {}

    def documents_to_sequences(self, documents):
        """
        Transforms each doc in documents in a sequence of integers.
        Only words in self.word_index will be taken into account.
        That is, Only words known by the tokenizer will be taken into account.

        # Arguments
            documents: A list of documents (strings).
        # Returns
            A list of sequences.
        """
        res = []

        for vect in self.texts_to_sequences_generator(documents):
            res.append(vect)
        return res

    def texts_to_sequences_generator(self, documents):
        """
        Transforms each doc in documents in a sequence of integers.
        Each item in texts can also be a list,
        in which case we assume each item of that list to be a token.
        Only words in self.word_index will be taken into account.
        That is, Only words known by the tokenizer will be taken into account.

        # Arguments
            texts: A list of documents (strings).
        # Yields
            Yields individual sequences.
        """

        for doc in documents:
            if self.char_level or isinstance(doc, list):
                seq = doc
            else:
                seq = doc.split(' ')

            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    vect.append(i)

                # elif self.oov_token is not None:
                #    i = self.word_index.get(self.oov_token)
                #    if i is not None:
                #        vect.append(i)
            yield vect
