# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : loss.py
# @time    : 16/12/2021 下午1:12
import numpy as np
from utils.param import one_hot


class CrossEntropyLoss(object):
    def __init__(self):
        self.grid = None

    def __call__(self, pre, y):
        """
        :param pre: pre_prob, shape like [n_sample, n_classes]
        :param y: target, a one-hot value, shape like [n_sample, n_classes]
        :return:
        """
        self.grid = pre.copy()
        pre = np.sum(y * pre, axis=1)
        self.grid -= y
        loss = np.mean(-np.log(pre+1.e-10))
        return loss

    def backward(self):
        return self.grid


class MSELoss(object):
    def __init__(self):
        self.grid = None

    def __call__(self, pre, y):
        """
        :param pre: pre_prob, shape like [n_sample, n_classes]
        :param y: target, not a one-hot, shape like [n_sample, n_output]
        :return:
        """
        loss = np.mean((pre - y)**2)
        self.grid = 2. * (pre - y)
        return loss

    def backward(self):
        return self.grid
