# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : binary.py
# @time    : 30/11/2021 下午8:53
import numpy as np


def confusion_matrix(prob, y, th=0.5):
    """
    binary classification confusion matrix.
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
    :param y: target
    :param th: the threshold of logic
    :return: TP, FP, TN, FN
    """
    logic = np.zeros_like(prob)
    logic[prob >= th] = 1
    tp = np.sum((y == 1) * (logic == 1))
    tn = np.sum((y == 0) * (logic == 0))
    fp = np.sum((y == 0) * (logic == 1))
    fn = np.sum((y == 1) * (logic == 0))
    return tp, fp, tn, fn


def recall(prob, y, th=0.5):
    """
    binary recall, focus on selecting more positive examples, tp / (tp + fn)
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
    :param y: target
    :param th: the threshold of logic
    :return: the value of recall
    """

    tp, _, _, fn = confusion_matrix(prob, y, th)
    return tp / (tp + fn)


def precision(prob, y):
    """
    binary precision, focus on selecting positive examples precisely, tp / (tp + fp)
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
    :param y: target
    :return: the value of precision
    """
    tp, fp, _, _ = confusion_matrix(prob, y)
    return tp / (tp + fp)


def accuracy(prob, y, th=0.5):
    """
    get the accuracy of the binary classification task. (tp + tn) / (tp + fp + tn + fn)
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
    :param y: target
    :param th: the threshold of logic
    :return: acc
    """
    tp, fp, tn, fn = confusion_matrix(prob, y, th)
    return (tp + tn) / (tp + fp + tn + fn)


def tpr(prob, y, th=0.5):
    """
    It is recall
    """
    return recall(prob, y, th)


def fpr(prob, y, th=0.5):
    """
    FPR = FP / (FP + TN)
    """
    _, fp, tn, _ = confusion_matrix(prob, y, th)
    return fp / (fp + tn)


def f_score(prob, y, beta=1.):
    """
    weighted sum between recall and precision, f_beta = [(1+beta**2)*p*r] / [(beta**2*p)+r]
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
    :param y: target
    :param beta:
    :return:
    """
    r = recall(prob, y)
    p = precision(prob, y)
    f_beta = ((1 + beta ** 2) * p * r) / ((beta ** 2 * p) + r)
    return f_beta


def roc(prob, y):
    """
    Receiver Operating Characteristic, X-coordinate is FPR, y-coordinate is TPR
    :return: _fpr(x), _tpr(y)
    """
    ths = np.sort(np.unique(prob))
    ths = ths[::-1]
    _tpr = []
    _fpr = []
    for th in ths:
        _tpr.append(tpr(prob, y, th))
        _fpr.append(fpr(prob, y, th))
    return _fpr, _tpr


def auc(_fpr, _tpr):
    """
    Calculate the area under the ROC curve,
    auc = summation i to n {[x_(i+1) - x_(i)] * 1/2*[y_(i+1) + y_(i)]}, except x_(i+1) = x_(i)
    :param _fpr: x
    :param _tpr: y
    :return: auc value
    """
    _auc = 0.
    for i in range(len(_fpr)-1):
        if _fpr[i+1] == _fpr[i]:
            continue
        xs = _fpr[i+1] - _fpr[i]
        ys = (_tpr[i+1] + _tpr[i]) / 2.
        _auc += (xs * ys)
    return _auc


def _for_test():
    Y = (np.concatenate([np.ones([50]), np.zeros([50])]))
    np.random.shuffle(Y)
    Prob = np.random.random([100])
    x, y = roc(Prob, Y)
    print(auc(x, y))
