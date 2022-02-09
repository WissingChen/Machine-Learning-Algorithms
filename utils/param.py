# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : param.py
# @time    : 18/12/2021 下午6:37
import numpy as np


def init_params(in_m, out_m, method='kaiming'):
    if method == "kaiming":
        return np.random.randn(in_m, out_m) * np.sqrt(12 / (in_m + out_m))
    elif method == 'xavier':
        return np.random.randn(in_m, out_m) * np.sqrt(6 / (in_m + out_m))
    elif method == 'none':
        return np.random.randn(in_m, out_m) * 1.e-3


def one_hot(label):
    label = label.astype(np.int8).reshape([-1])
    dst = np.zeros((label.size, label.max() + 1))
    dst[np.arange(label.size), label] = 1
    return dst
