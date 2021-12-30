# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : regression.py
# @time    : 30/11/2021 下午8:53
import numpy as np


def mse(pre, y):
    """
    MSE - Mean Squared Error
    """
    return np.mean((pre - y) ** 2)


def mae(pre, y):
    """
    MAE - Mean Absolute Error
    """
    return np.mean(np.abs(pre - y))


def pearson_corr(pre, y):
    m, n = y.shape
    corr = 0.
    for i in range(n):
        corr += np.corrcoef(pre[:, i], y[:, i])[0, 1]
    return corr / n


def spearman_corr(pre, y):
    m, n = y.shape

    def _rank_corr(_a, _b, _n):
        upper = 6 * np.sum((_a - _b) ** 2)
        down = _n * (_n ** 2 - 1.0)
        return 1. - (upper / down)

    corr = 0.
    for i in range(n):
        a = pre[:, i].argsort()
        b = y[:, i].argsort()
        corr += _rank_corr(a, b, m)
    return corr / n
