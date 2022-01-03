# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : Ridge.py
# @time    : 1/12/2021 下午6:54
import numpy as np
from ..base import Base
from utils.metric.regression import mse


class Ridge(Base):
    def __init__(self, alpha=.1):
        Base.__init__(self)
        self.w = None
        self.alpha = alpha

    def fit(self, x, y):
        m, n = x.shape
        y = y.reshape([-1, 1])
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        self.w = np.linalg.inv(np.dot(x.T, x) + self.alpha * np.eye(n+1)).dot(x.T).dot(y)
        self._score = mse(np.dot(x, self.w), y)

    def predict(self, x):
        m = x.shape[0]
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        return np.dot(x, self.w)

    def score(self):
        return self._score
