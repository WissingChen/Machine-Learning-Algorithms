# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _sgd.py
# @time    : 4/1/2022 下午7:27
import numpy as np
from utils.param import init_params
from utils.functional import Identity


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
    def __init__(self, loss='log', penalty='l1', lr=1.e-3):
        super(SGDClassifier, self).__init__(loss, penalty, lr)
        pass

    def get_loss(self, loss):
        if loss == 'log':
            self.loss = mse
        elif loss == 'hinge':
            # TODO (soft-interval) linear support vector machines
            pass
        elif loss == 'modified_huber':
            # TODO smoothed hinge loss
            pass

    def fit(self, x, y, iteration=100):
        """
        :param x: [n_sample, n_feature]
        :param y: [n_sample, n_classes]
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

    def predict(self, x):
        z = x.dot(self.w) + self.bias
        return 1 / (1 + np.exp(-z))


def mse(pre, y):
    l = np.mean((pre - y)**2)
    dl = 2*(pre - y)
    return l, dl


# todo not finish yet
def log(pre, y):
    pre()