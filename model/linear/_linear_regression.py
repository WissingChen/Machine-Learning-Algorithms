# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _linear_regression.py
# @time    : 1/12/2021 上午10:02
import numpy as np
from utils.metric.regression import mse


class LinearRegression(object):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    In this instance, I use QR decomposition to calculate the analytical solution of linear regression.
    """
    def __init__(self):
        self.w = None
        self._score = None

    def fit(self, x, y):
        """
        Fit Linear Regression model.

        :param x: input data with shape [n_sample, n_feature]
        :param y: target with shape [n_sample, n_output] or [n_sample,]
        """
        m, n = x.shape
        y = y.reshape([m, -1])
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        q, r = np.linalg.qr(x)
        self.w = np.linalg.inv(r).dot(q.T).dot(y)
        # use MSE
        self._score = mse(np.dot(x, self.w), y)

    def predict(self, x):
        m = x.shape[0]
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        return np.dot(x, self.w)

    def score(self):
        return self._score
