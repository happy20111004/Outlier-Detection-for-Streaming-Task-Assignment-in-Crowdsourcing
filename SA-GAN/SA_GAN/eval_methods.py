# -*- coding: utf-8 -*-
import numpy as np


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN



def calc_seq(score, label, threshold):
    """
    Calculate f1 score for a score sequence
    """

    predict = score > threshold


    return calc_point2point(predict, label)


def bf_search(score, label):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """

    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(98):
        threshold  = 0.01 * i + 0.01
        target = calc_seq(score, label, threshold)
        if target[0] > m[0]:
            m_t = threshold
            m = target

    return m, m_t


