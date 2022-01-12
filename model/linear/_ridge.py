# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : Ridge.py
# @time    : 1/12/2021 下午6:54
import numpy as np
from scipy.sparse import linalg as sp_linalg
from ..base import Base
from utils.metric.regression import mse
from utils.metric.binary import _acc


class Ridge(Base):
    """
    Linear least squares with l2 regularization.

    Minimizes the objective function:

    ||y - Xw||^2_2 + alpha * ||w||^2_2

    This model solves a regression model where the loss function is
    the linear least squares function and regularization is given by
    the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
    This estimator has built-in support for multi-variate regression
    (i.e., when y is a 2d-array of shape (n_samples, n_targets)).

    Parameter
    ---------
    alpha: l2 coefficient   \n
    solver: choice in ['none', 'lsqr']

    """
    def __init__(self, alpha=.1, solver='none'):
        Base.__init__(self)
        self.w = None
        self.alpha = alpha
        self.solver = solver

    def fit(self, x, y):
        """
        Fit Ridge Regression model.

        Parameters
        ----------
        x : input data with shape [n_sample, n_feature]
        y : target with shape [n_sample, n_output] or [n_sample,]

        """
        m, n = x.shape
        y = y.reshape([m, -1])
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        if self.solver == 'lsqr':
            self.w = _lsqr(x, y, self.alpha)
        elif self.solver == 'none':
            self.w = np.linalg.inv(np.dot(x.T, x) + self.alpha * np.eye(n+1)).dot(x.T).dot(y)
        # use mse
        self._score = mse(np.dot(x, self.w), y)

    def predict(self, x):
        m = x.shape[0]
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        return np.dot(x, self.w)

    def score(self):
        return self._score


def _lsqr(x, y, alpha):
    """
    References
    ----------
    .. [1] C. C. Paige and M. A. Saunders (1982a).
           "LSQR: An algorithm for sparse linear equations and
           sparse least squares", ACM TOMS 8(1), 43-71.
    .. [2] C. C. Paige and M. A. Saunders (1982b).
           "Algorithm 583.  LSQR: Sparse linear equations and least
           squares problems", ACM TOMS 8(2), 195-209.
    .. [3] M. A. Saunders (1995).  "Solution of sparse rectangular
           systems using LSQR and CRAIG", BIT 35, 588-604.
    """
    n, n = x.shape
    w = np.zeros([n, y.shape[1]])

    sqrt_alpha = np.sqrt(alpha)

    for i in range(y.shape[1]):
        y_column = y[:, i]
        info = sp_linalg.lsqr(x, y_column, damp=sqrt_alpha)
        w[:, i] = info[0]

    return w


class RidgeClassifier(Ridge):
    """
    Only for binary classification

    Parameter
    ---------
    alpha: l2 coefficient   \n
    solver: choice in ['none', 'lsqr']
    """
    def __init__(self, alpha=.1, solver='none'):
        Ridge.__init__(self, alpha=alpha, solver=solver)

    @staticmethod
    def cvt_input(y):
        y_hat = y.copy()
        y_hat[y_hat == 0] = -1
        return y_hat

    def fit(self, x, y):
        y_hat = self.cvt_input(y)
        Ridge.fit(self, x, y_hat)
        # use acc
        self._score = _acc(self.predict(x), y)

