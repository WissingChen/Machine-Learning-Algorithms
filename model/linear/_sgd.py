# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _sgd.py
# @time    : 4/1/2022 下午7:27
import numpy as np
from utils.param import init_params
from utils.functional import Identity, softmax


class SGDBase(object):
    def __init__(self, loss, alpha=1., penalty='none', lr=1.e-4):
        self.w = None
        self.bias = None
        self.loss = None
        self.lr = lr
        self.penalty = penalty
        self.alpha = alpha
        self._score = None

        self.get_loss(loss)

    def fit(self, x, y, iteration):
        pass

    def score(self):
        return self._score

    def fit_one_loop(self, x, y):
        pre = self.predict(x)
        loss, gl = self.loss(pre, y)
        if self.penalty == 'l1':
            gw = np.dot(x.T, gl) + self.alpha * np.sign(self.w)
        elif self.penalty == 'l2':
            gw = np.dot(x.T, gl) + self.alpha * self.w
        elif self.penalty == 'none':
            gw = np.dot(x.T, gl)
        else:
            raise ValueError
        gb = np.sum(gl, axis=0, keepdims=True)
        self.w = self.w - self.lr * gw
        self.bias = self.bias - self.lr * gb

    def get_loss(self, loss):
        self.loss = Identity()

    def predict(self, x):
        return x.dot(self.w) + self.bias


class SGDRegression(SGDBase):
    def __init__(self, loss='mse', alpha=1., penalty='l1', lr=1.e-3):
        super(SGDRegression, self).__init__(loss, alpha, penalty, lr)

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
    def __init__(self, loss='ce', alpha=1., penalty='l1', lr=1.e-3):
        super(SGDClassifier, self).__init__(loss, alpha, penalty, lr)
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

    def fit_one_loop(self, x, y):
        pre = self.predict(x, 'prob')
        loss, gl = self.loss(pre, y)
        if self.penalty == 'l1':
            gw = np.dot(x.T, gl) + self.alpha * np.sign(self.w)
        elif self.penalty == 'l2':
            gw = np.dot(x.T, gl) + self.alpha * self.w
        elif self.penalty == 'none':
            gw = np.dot(x.T, gl)
        else:
            raise ValueError
        gb = np.sum(gl, axis=0, keepdims=True)
        self.w = self.w - self.lr * gw
        self.bias = self.bias - self.lr * gb

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
        self._score, _ = ce(self.predict(x, value_type='prob'), y)

    def predict(self, x, value_type='class'):
        z = x.dot(self.w) + self.bias
        y = softmax(z, dim=-1)
        if value_type == 'class':
            return np.argmax(y, axis=1)
        elif value_type == 'prob':
            return y
        else:
            raise ValueError


def mse(pre, y):
    l = np.mean((pre - y) ** 2)
    gl = 2 * (pre - y)
    return l, gl


def ce(pre, y):
    y = y.reshape(-1)
    n_classes = len(np.unique(y))
    one_hot = np.zeros_like(pre)
    gl = pre.copy()
    for i in range(n_classes):
        one_hot[y == i, i] = 1
    pre = np.sum(one_hot * pre, axis=1)
    gl -= one_hot
    l = np.mean(-np.log(pre + 1.e-10))
    return l, gl


def huber(true, pred, delta):
    loss = np.where(np.abs(true - pred) < delta, 0.5 * ((true - pred) ** 2),
                    delta * np.abs(true - pred) - 0.5 * (delta ** 2))
    return np.sum(loss)
