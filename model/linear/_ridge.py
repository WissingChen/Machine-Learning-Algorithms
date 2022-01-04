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


class Ridge(Base):
    def __init__(self, alpha=.1, solver='none'):
        Base.__init__(self)
        self.w = None
        self.alpha = alpha
        self.solver = solver

    def fit(self, x, y):
        m, n = x.shape
        y = y.reshape([-1, 1])
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        if self.solver == 'lsqr':
            self.w = _lsqr(x, y, self.alpha)
        elif self.solver == 'none':
            self.w = np.linalg.inv(np.dot(x.T, x) + self.alpha * np.eye(n+1)).dot(x.T).dot(y)
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
