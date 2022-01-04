# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _lasso.py
# @time    : 4/1/2022 下午7:08
import numpy as np
from ..base import Base
from utils.metric.regression import mse


class Lasso(Base):
    def __init__(self, alpha=.1, solver='none'):
        Base.__init__(self)
        self.w = None
        self.alpha = alpha
        self.solver = solver

    def fit(self, x, y):
        m, n = x.shape
        y = y.reshape([-1, 1])
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        if self.solver == 'cd':
            # self.w = _lsqr(x, y, self.alpha)
            pass
        elif self.solver == 'none':
            self.w = _ols(x, y, self.alpha)
        self._score = mse(np.dot(x, self.w), y)

    def predict(self, x):
        m = x.shape[0]
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        return np.dot(x, self.w)

    def score(self):
        return self._score


def _ols(x, y, alpha):
    m, n = x.shape
    w_ols = x.T.dot(y)
    w = np.sign(w_ols) * np.maximum(np.abs(w_ols - m*alpha/2), 0)
    return w


def _cd():
    pass
