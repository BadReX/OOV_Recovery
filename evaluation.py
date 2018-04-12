# -*- coding: utf-8 -*-
"""
A module for the computing evaluation metrics.
"""

import numpy as np


def get_class_predictions(y_test, nn_preds, threshold=0.5, topK=None):
    """
    Given a test set, return predicted document class labels.

    # Arguments
        y_test: a numpy array of binarized labels (dtype: numpy.ndarray)
        nn_preds: a numpy array of sigoid predictions (dtype: numpy.ndarray)
        threshold: a threshold value to consider (dtype: float)
        topK: an integer for K top activation values (dtype: int)

    # Returns
        pred_labels: an ordered list (by probablility) of predicted labels
    """

    pred_labels = list()

    for idx in range(len(nn_preds)):
        if topK:
            # get the top K predictions based on the activation values of the output layer
            preds = list(reversed(nn_preds[idx].ravel().argsort()[-topK:]))

        else:
            # get the predictions based on the given threshold
            preds = np.where(nn_preds[idx] > threshold)[0].tolist()

        trues = np.where(y_test[idx] == 1)[0].tolist()
        #trues = mlb.inverse_transform(y_test[idx])

        pred_labels.append(preds)

    return pred_labels


def compute_metrics(predictions, actual_labels):
    """
    Given ground truth classes of test set and predictions,
    compute and return classification performance metrics (P, R, and F)

    # Arguments:
        zipped_labels: a zip object of (true_labels, predicted_labels)

    # Returns
        Micro-averaged precision, recall, and F-measure
    """

    # test shapes of the two arrays
    assert y_test.shape == nn_preds.shape, "array shape mismatch."
    print("Computing performance measures ...")
    total_TPs = 0
    total_FPs = 0
    total_FNs = 0
    total_preds = 0
    total_trues = 0

    for (pred_labels, true_labels) in zip(predictions, actual_labels):

        TPs = set(true_labels).intersection(pred_labels)
        FPs = set(pred_labels) - set(true_labels)
        FNs = set(true_labels) - TPs

        total_TPs += len(TPs)
        total_FPs += len(FPs)
        total_FNs += len(FNs)

    # micro-averaged statisitics
    avgP = total_TPs/(total_TPs + total_FPs)
    avgR = total_TPs/(total_TPs + total_FNs)

    # TODO: solve the division by zero exception
    try:
        F_score = (2*total_TPs)/(2*total_TPs + total_FPs + total_FNs)
    except:
        F_score = float('-inf')

    return (avgP, avgR, F_score)
