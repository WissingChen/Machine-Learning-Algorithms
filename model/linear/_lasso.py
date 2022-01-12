# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _lasso.py
# @time    : 4/1/2022 下午7:08
import numpy as np
from ..base import Base
from utils.metric.regression import mse
from utils.param import init_params


class Lasso(Base):
    """
    Linear Model trained with L1 prior as regularizer (aka the Lasso)

    The optimization objective for Lasso is::

        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    Technically the Lasso model is optimizing the same objective function as
    the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).

    :param alpha: l1 value
    :param solver: 'none', 'cd', 'lars'
                    'none': only use Ordinary Least Squares
                    'cd': Coordinate Descent
                    'lars': Least Angle Regression
    """
    def __init__(self, alpha=.1, solver='none'):
        Base.__init__(self)
        self.w = None
        self.alpha = alpha
        self.solver = solver

    def fit(self, x, y):
        """
        Fit Lasso Regression model.

        Parameters
        ----------
        x : input data with shape [n_sample, n_feature]
        y : target with shape [n_sample, n_output] or [n_sample,]

        """
        m, n = x.shape
        y = y.reshape([m, -1])
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        if self.solver == 'cd':
            self.w = _cd(x, y, self.alpha)
        elif self.solver == 'none':
            self.w = _ols(x, y, self.alpha)
        elif self.solver == 'lars':
            self.w = _lars(x, y, self.alpha)
        self._score = mse(np.dot(x, self.w), y)

    def predict(self, x):
        m = x.shape[0]
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        return np.dot(x, self.w)

    def score(self):
        return self._score


def _ols(x, y, alpha):
    """
    Ordinary Least Squares
    """
    m, n = x.shape
    # w_ols isn`t 'x.T.dot(y)' ?
    w_ols = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    w = np.sign(w_ols) * np.maximum(np.abs(w_ols - m*alpha/2), 0)
    return w


def _cd(x, y, alpha, max_iter=100, tol=1e-5):
    """
    Coordinate Descent
    """
    m, n = x.shape
    w = init_params(n, y.shape[1])
    old_w = np.zeros_like(w)
    for _iter in range(max_iter):
        for k in range(n):
            old_w[k] = w[k]

            y_predict = np.dot(x, w)
            rk = np.dot(x[:, k], y - y_predict + x[:, k] * w[k])
            rk = np.mean(rk)
            zk = np.linalg.norm(x[:, k], ord=2) ** 2
            zk = np.mean(zk)
            w_k = np.maximum(rk - alpha, 0) - np.maximum(-rk - alpha, 0)
            w[k] = w_k / (1.0 * zk)

        # check conditions of convergence
        w_updates = np.abs(w - old_w)
        if np.max(w_updates) < tol:
            break
    return w


def _lars(x, y, alpha):
    # todo LARS for Lasso
    pass
