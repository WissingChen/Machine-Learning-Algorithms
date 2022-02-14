# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _svm.py
# @time    : 4/2/2022 下午8:11
import numpy as np
from utils.param import init_params
from utils.metric.binary import acc_v2
from utils.metric.regression import mse


class BaseSVM(object):
    def __init__(self, C=1.):
        self.C = C
        self.w = None
        self.b = 0
        self.alpha = None

        self.turn = 1
        self._score = None

    def fit(self, x, y, tol=1.e-3):
        pass

    def _predict(self, x):
        pre = x.dot(self.w) + self.b
        return pre * self.turn

    def predict(self, x):
        return self._predict(x)

    def score(self):
        return self._score


class SVC(BaseSVM):
    def __init__(self, C=1.):
        super(SVC, self).__init__(C=C)
        pass

    def fit(self, x, y, tol=1.e-3):
        m, n = x.shape
        y_hat = y.copy()
        y_hat[y_hat == 0] = -1
        self.alpha = init_params(m, 1)
        self.w = np.sum(self.alpha * y_hat * x, axis=0)
        for _ in range(m):
            old_w = self.w
            ai = np.argmax(self.alpha)
            aj = np.argmin(self.alpha)
            while True:
                old_l = self.alpha[ai]
                l_y = np.sum(self.alpha * y_hat)
                l_i = self.alpha[ai] * y_hat[ai]
                l_j = self.alpha[aj] * y_hat[aj]
                c = -l_y + l_i + l_j
                self.alpha[ai] = (c - 2 / (y_hat[ai] * x[ai].dot(x[aj].T))) / (2 * y_hat[ai])
                # self.L[ai] = c / (2 * y_hat[ai] * y_hat[aj] * x[ai].dot(x[aj].T)) ????
                self.alpha[aj] = (c - self.alpha[aj] * y_hat[ai]) * y_hat[aj]
                if self.alpha[ai] >= 0 or old_l == self.alpha[ai]:
                    break

            self.w = np.sum(self.alpha * y_hat * x, axis=0)
            w_updates = np.abs(self.w - old_w)
            if np.max(w_updates) < tol:
                break

        b = y_hat - x.dot(self.w)
        self.b = np.mean(b[self.alpha.reshape(-1) > 0])
        self._score = acc_v2(self.predict(x).reshape(-1), y.reshape(-1))
        if self._score <= .2:
            self.turn = -1
            self._score = 1 - self._score

    def predict(self, x):
        pre = np.sign(self._predict(x))
        pre[pre == -1] = 0
        return pre


class SVR(BaseSVM):
    def __init__(self, C=1., epsilon=0.1):
        super(SVR, self).__init__(C=C)
        self.epsilon = epsilon
        pass

    def fit(self, x, y, tol=1.e-3):
        m, n = x.shape
        y_hat = y.copy()
        self.alpha = init_params(m, 2)  # [a, a_hat]
        self.w = np.sum((self.alpha[:, 1:] - self.alpha[:, :1]) * x, axis=0)
        for _ in range(m):
            old_w = self.w
            ai = np.argmax(self.alpha[:, 0])
            aj = np.argmin(self.alpha[:, 0])

            # optimize alpha
            while True:
                old_a = self.alpha[ai, 0]
                a_y = np.sum(self.alpha[:, 1] - self.alpha[:, 0])
                c = -a_y + (self.alpha[ai, 1] - self.alpha[ai, 0]) + (self.alpha[aj, 1] - self.alpha[aj, 0])
                self.alpha[ai: 0] = self.alpha[ai: 1] - c / 2. + (self.epsilon + y_hat[ai]) / x[ai].dot(x[aj].T)
                self.alpha[aj: 0] = self.alpha[aj: 1] - c + self.alpha[ai: 1] - self.alpha[ai: 0]
                t_a = self.alpha[ai, 0]
                if 0 <= self.alpha[ai, 0] <= self.C or old_a == self.alpha[ai, 0]:
                    break

            # optimize alpha_hat
            ai = np.argmax(self.alpha[:, 1])
            aj = np.argmin(self.alpha[:, 1])
            while True:
                old_a = self.alpha[ai, 1]
                a_y = np.sum(self.alpha[:, 1] - self.alpha[:, 0])
                c = -a_y + (self.alpha[ai, 1] - self.alpha[ai, 0]) + (self.alpha[aj, 1] - self.alpha[aj, 0])
                self.alpha[ai: 1] = self.alpha[ai: 0] + c / 2. + (self.epsilon - y_hat[ai]) / x[ai].dot(x[aj].T)
                self.alpha[aj: 1] = self.alpha[aj: 0] + c - self.alpha[ai: 1] + self.alpha[ai: 0]
                if 0 <= self.alpha[ai, 1] <= self.C or old_a == self.alpha[ai, 1]:
                    break

            self.w = np.sum((self.alpha[:, 1:] - self.alpha[:, :1]) * x, axis=0)
            w_updates = np.abs(self.w - old_w)
            if np.max(w_updates) < tol:
                break

        b = y_hat + self.epsilon - x.dot(self.w)
        self.b = np.mean(b[self.alpha[:, 1] != self.alpha[:, 0]])
        self._score = mse(self.predict(x), y)
