# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _linear_regression.py
# @time    : 1/12/2021 上午10:02
import numpy as np
from ..base import Base
from utils.metric_regression import mse


class LinearRegression(Base):
    def __init__(self):
        Base.__init__(self)
        self.w = None

    def fit(self, x, y):
        m, n = x.shape
        y = y.reshape([-1, 1])
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        q, r = np.linalg.qr(x)
        self.w = np.linalg.inv(r).dot(q.T).dot(y)
        self._score = mse(np.dot(x, self.w), y)

    def predict(self, x):
        m = x.shape[0]
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        return np.dot(x, self.w)

    def score(self):
        pass
