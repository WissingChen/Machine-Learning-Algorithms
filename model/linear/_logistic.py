# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _logistic.py
# @time    : 1/12/2021 下午7:16
import numpy as np
from utils.param import init_params, one_hot
from utils.metric.multi import acc_v2 as acc_v2_m
from utils.metric.binary import acc_v2
from utils.functional import sigmoid


class LogisticRegression(object):
    def __init__(self, penalty='l2', solver='sgd', ):
        self.w = None
        self.solver = solver
        self._score = None

    def fit(self, x, y):
        """
        Fit Logistic Regression model.

        Parameters
        ----------
        x : input data with shape [n_sample, n_feature]
        y : target with shape [n_sample, 1] or [n_sample,]

        """
        m, n = x.shape
        y = y.reshape(-1)
        n_classes = len(np.unique(y))
        _x = np.concatenate([np.ones([m, 1]), x], axis=1)
        if self.solver == 'cd':
            self.w = _cd(_x, y)
        elif self.solver == 'sgd':
            self.w = _sgd(_x, y)

        if n_classes == 2:
            self._score = acc_v2(self.predict(x).reshape(-1), y.reshape(-1))
        else:
            self._score = acc_v2_m(self.predict(x).reshape(-1), y.reshape(-1))

    def _predict(self, x):
        m = x.shape[0]
        x = np.concatenate([np.ones([m, 1]), x], axis=1)
        return np.dot(x, self.w)

    def predict(self, x, value='class'):
        pre = self._predict(x)
        if value == 'class':
            return np.argmax(pre, axis=1)
        elif value == 'prob':
            return sigmoid(pre)

    def score(self):
        return self._score


def _cd(x, y, alpha=1., max_iter=100, tol=1e-5):
    m, n = x.shape
    n_classes = len(np.unique(y.reshape(-1)))
    w = init_params(n, n_classes)
    old_w = np.zeros_like(w)
    y_hat = one_hot(y)
    for _iter in range(max_iter):
        for k in range(n):
            old_w[k] = w[k]
            z = x.dot(w)
            # todo
        # check conditions of convergence
        w_updates = np.abs(w - old_w)
        if np.max(w_updates) < tol:
            break
    return w


def _sgd(x, y, lr=1.e-5, alpha=1., momentum=0.9, max_iter=1000, tol=1e-4):
    is_max_iter = True
    m, n = x.shape
    n_classes = len(np.unique(y.reshape(-1)))
    w = init_params(n, n_classes)
    velocity = 0
    # old_w = np.zeros_like(w)
    y_hat = one_hot(y)
    _index = np.arange(m)
    # batch_size = int(m/10)
    for _iter in range(max_iter):
        # np.random.shuffle(_index)
        for _batch in range(m):
            old_w = w.copy()
            # temp_index = _index[_batch * batch_size: (_batch + 1) * batch_size]
            train_x = x[_index]
            train_y = y_hat[_index]
            z = train_x.dot(w)
            px = sigmoid(z)
            gx = train_x.T.dot((px - train_y)) + alpha * w

            velocity = lr * gx + momentum * velocity
            w -= velocity
            # check conditions of convergence
            w_updates = np.abs(w - old_w)
            if np.max(w_updates) < tol:
                is_max_iter = False
                break
    return w

