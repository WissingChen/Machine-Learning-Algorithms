# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _sgd.py
# @time    : 4/1/2022 下午7:27
import numpy as np
from utils.param import init_params
from utils.functional import Identity


# TODO still not add l1/l2
class SGDBase(object):
    def __init__(self, loss, penalty, lr):
        self.w = None
        self.bias = None
        self.loss = None
        self.lr = lr
        self.penalty = penalty
        self._score = None

        self.get_loss(loss)

    def fit(self, x, y, iteration):
        pass

    def score(self):
        return self._score

    def fit_one_loop(self, x, y):
        pre = self.predict(x)
        loss, dl = self.loss(pre, y)
        dw = np.dot(x.T, dl)
        db = np.sum(dl, axis=0, keepdims=True)
        self.w = self.w - self.lr * dw
        self.bias = self.bias - self.lr * db

    def get_loss(self, loss):
        self.loss = Identity()

    def predict(self, x):
        return x.dot(self.w) + self.bias


class SGDRegression(SGDBase):
    def __init__(self, loss='mse', penalty='l1', lr=1.e-3):
        super(SGDRegression, self).__init__(loss, penalty, lr)

    def get_loss(self, loss):
        if loss == 'mse':
            self.loss = mse
        elif loss == 'huber':
            # TODO Huber loss for robust regression
            pass
        elif loss == 'epsilon_insensitive':
            # TODO  linear Support Vector Regression
            pass

    def fit(self, x, y, iteration=100):
        """
        :param x: [n_sample, n_feature]
        :param y: [n_sample, n_output]
        :param iteration:
        :return:
        """
        m, n = x.shape
        # init params
        self.w = init_params(n, y.shape[1])
        self.bias = np.zeros([1, y.shape[1]])
        for iter in range(iteration):
            self.fit_one_loop(x, y)
        self._score, _ = mse(self.predict(x), y)


class SGDClassifier(SGDBase):
    def __init__(self, loss='ce', penalty='l1', lr=1.e-3):
        super(SGDClassifier, self).__init__(loss, penalty, lr)
        self.n_classes = None

    def get_loss(self, loss):
        if loss == 'ce':
            self.loss = ce
        elif loss == 'hinge':
            # TODO (soft-interval) linear support vector machines
            pass
        elif loss == 'modified_huber':
            # TODO smoothed hinge loss
            pass

    def fit(self, x, y, iteration=1000):
        """
        :param x: [n_sample, n_feature]
        :param y: [n_sample, ]
        :param iteration:
        :return:
        """
        m, n = x.shape
        y = y.reshape(-1)
        self.n_classes = len(np.unique(y))
        # init params
        self.w = init_params(n, self.n_classes)
        self.bias = np.zeros([1, self.n_classes])
        for iter in range(iteration):
            self.fit_one_loop(x, y)
        self._score, _ = ce(self.predict(x), y)

    def predict(self, x):
        z = x.dot(self.w) + self.bias
        _max = np.max(z, axis=1, keepdims=True)  # returns max of each row and keeps same dims
        e_z = np.exp(z - _max)  # subtracts each row with its max value
        _sum = np.sum(e_z, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
        y = e_z / _sum
        return y


def mse(pre, y):
    l = np.mean((pre - y)**2)
    dl = 2*(pre - y)
    return l, dl


def ce(pre, y):
    y = y.reshape(-1)
    n_classes = len(np.unique(y))
    one_hot = np.zeros_like(pre)
    dl = pre.copy()
    for i in range(n_classes):
        one_hot[y == i, i] = 1
    pre = np.sum(one_hot * pre, axis=1)
    dl -= one_hot
    l = np.mean(-np.log(pre + 1.e-10))
    return l, dl
