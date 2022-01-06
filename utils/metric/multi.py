# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : multi.py
# @time    : 17/12/2021 下午2:39
import numpy as np
from utils.metric.binary import roc, auc


def confusion_matrix(prob, y):
    """
    mu classification confusion matrix.
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
                    a confusion matrix, [target, prediction]
    :param y: target
    :param th: the threshold of logic
    :return: tp, fp, tn, fn for each classes
    """
    n = prob.shape[1]
    cm = np.zeros([n, n])
    logic = np.argmax(prob, axis=1)
    for i in range(prob.shape[0]):
        cm[int(y[i]), logic[i]] += 1

    tp = np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    return tp, fp, tn, fn


def recall(prob, y, mode='macro'):
    """
    binary recall, focus on selecting more positive examples, tp / (tp + fn)
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
    :param y: target
    :param th: the threshold of logic
    :return: the value of recall
    """

    tp, _, _, fn = confusion_matrix(prob, y)
    _recall = tp / (tp + fn)
    macro = _recall.mean()
    if mode == 'all':
        return _recall
    elif mode == 'macro':
        return macro
    elif mode == 'micro':
        micro = tp.sum() / (tp.sum() + fn.sum())
        return micro


def precision(prob, y, mode='macro'):
    """
    binary precision, focus on selecting positive examples precisely, tp / (tp + fp)
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
    :param y: target
    :return: the value of precision
    """
    tp, fp, _, _ = confusion_matrix(prob, y)
    _precision = tp / (tp + fp)
    macro = _precision.mean()
    if mode == 'all':
        return _precision
    elif mode == 'macro':
        return macro
    elif mode == 'micro':
        micro = tp.sum() / (tp.sum() + fp.sum())
        return micro


def accuracy(prob, y, mode='macro'):
    """
    get the accuracy of the binary classification task. (tp + tn) / (tp + fp + tn + fn)
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
    :param y: target
    :param th: the threshold of logic
    :return: acc
    """
    tp, fp, tn, fn = confusion_matrix(prob, y)
    _acc = (tp + tn) / (tp + fp + tn + fn)
    macro = _acc.mean()
    if mode == 'all':
        return _acc
    elif mode == 'macro':
        return macro
    elif mode == 'micro':
        micro = (tp.sum() + tn.sum()) / (tp.sum() + fp.sum() + tn.sum() + fn.sum())
        return micro


def tpr(prob, y, mode='macro'):
    """
    It is recall
    """
    return recall(prob, y, mode)


def fpr(prob, y, mode='macro'):
    """
    FPR = FP / (FP + TN)
    """
    _, fp, tn, _ = confusion_matrix(prob, y)
    _fpr = fp / (fp + tn)
    macro = _fpr.mean()
    if mode == 'all':
        return _fpr
    elif mode == 'macro':
        return macro
    elif mode == 'micro':
        micro = fp.sum() / (fp.sum() + tn.sum())
        return micro


def f_score(prob, y, beta=1., mode='macro'):
    """
    weighted sum between recall and precision, f_beta = [(1+beta**2)*p*r] / [(beta**2*p)+r]
    :param mode:
    :param prob: the prob prediction, you can also use logic straightly. And the input is one dim.
    :param y: target
    :param beta:
    :return:
    """
    r = recall(prob, y, mode=mode)
    p = precision(prob, y, mode=mode)
    f_beta = ((1 + beta ** 2) * p * r) / ((beta ** 2 * p) + r)
    return f_beta


def roc_auc(prob, y, mode='macro'):
    """
    Calculate the area under the ROC curve,
    auc = summation i to n {[x_(i+1) - x_(i)] * 1/2*[y_(i+1) + y_(i)]}, except x_(i+1) = x_(i)
    :param mode:
    :param prob:
    :param y: should be [n_sample,]
    :return: roc={fpr, tpr}, auc value
    """
    n_classes = prob.shape[1]
    label = np.zeros_like(prob)
    y = y.reshape(-1)
    # 计算每一个类别的
    _roc_auc = dict()
    _fpr = dict()
    _tpr = dict()
    for i in range(n_classes):
        label[y == i, i] = 1
        _fpr[i], _tpr[i] = roc(label[:, i], prob[:, i])
        _roc_auc[i] = auc(_fpr[i], _tpr[i])

    if mode == 'all':
        return _fpr, _tpr, _roc_auc

    elif mode == 'macro':
        # 首先收集所有的假正率
        all_fpr = np.unique(np.concatenate([_fpr[i] for i in range(n_classes)]))

        # 然后在此点内插所有ROC曲线
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, _fpr[i], _tpr[i])

        # 最终计算平均和ROC
        mean_tpr /= n_classes

        return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    elif mode == 'micro':
        micro_fpr, micro_tpr = roc(label.ravel(), prob.ravel())
        return micro_fpr, micro_tpr, auc(micro_fpr, micro_tpr)


def _for_test():
    Y = (np.concatenate([np.ones([50]), np.zeros([50]), np.ones([50])*2]))
    np.random.shuffle(Y)
    Prob = np.random.random([150, 3])
    print(roc_auc(Prob, Y)[2])


