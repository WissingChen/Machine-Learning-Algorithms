# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _elasticnet.py
# @time    : 4/1/2022 下午10:27
import numpy as np
from utils.metric.regression import mse
from utils.param import init_params


class ElasticNet(object):
    """
    Linear regression with combined L1 and L2 priors as regularizer.

    Minimizes the objective function::

            1 / (2 * n_samples) * ||y - Xw||^2_2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    If you are interested in controlling the L1 and L2 penalty
    separately, keep in mind that this is equivalent to::

            a * ||w||_1 + 0.5 * b * ||w||_2^2

    where::

            alpha = a + b and l1_ratio = a / (a + b)

    The parameter l1_ratio corresponds to alpha in the glmnet R package while
    alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio
    = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,
    unless you supply your own sequence of alpha.
    """
    def __init__(self, alpha=1., l1_ratio=.5, solver='cd'):
        self.w = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.solver = solver
        self._score = None

    def fit(self, x, y):
        """
        Fit ElasticNet Regression model.

        :param x: input data with shape [n_sample, n_feature]
        :param y: target with shape [n_sample, n_output] or [n_sample,]

        """
        m, n = x.shape
        y = y.reshape([m, -1])
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        self.w = _cd(x, y, alpha=self.alpha, l1_ratio=self.l1_ratio)
        # use mse
        self._score = mse(np.dot(x, self.w), y)

    def _predict(self, x):
        m = x.shape[0]
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        return np.dot(x, self.w)

    def predict(self, x):
        return self._predict(x)

    def score(self):
        return self._score


def _cd(x, y, alpha, l1_ratio, max_iter=100, tol=1e-5):
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
            zk = np.mean(zk) + alpha * (1 - l1_ratio)

            w_k = np.maximum(rk - alpha*l1_ratio, 0) - np.maximum(-rk - alpha*l1_ratio, 0)
            w[k] = w_k / (1.0 * zk)

        # check conditions of convergence
        w_updates = np.abs(w - old_w)
        if np.max(w_updates) < tol:
            break
    return w
