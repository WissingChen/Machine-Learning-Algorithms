# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _svm.py
# @time    : 4/2/2022 下午8:11
import numpy as np
from utils.param import init_params
from utils.metric.binary import acc_v2


class BaseSVM(object):
    def __init__(self):
        self.w = None
        self.b = 0
        self.L = None
        self.C = None
        self.turn = 1

        self._score = None

    def fit(self, x, y, tol=1.e-3):
        m, n = x.shape
        y_hat = y.copy()
        y_hat[y_hat == 0] = -1
        self.L = init_params(m, 1)
        self.w = np.sum(self.L * y_hat * x, axis=0)
        for _ in range(m):
            old_w = self.w
            li = np.argmax(self.L)
            lj = np.argmin(self.L)
            while True:
                old_l = self.L[li]
                l_y = np.sum(self.L * y_hat)
                l_i = self.L[li] * y_hat[li]
                l_j = self.L[lj] * y_hat[lj]
                c = -l_y + l_i + l_j
                self.L[li] = c / (2 * y_hat[li] * y_hat[lj] * x[li].dot(x[lj].T))
                self.L[lj] = (c - self.L[lj] * y_hat[li]) * y_hat[lj]
                if self.L[li] >= 0 or old_l == self.L[li]:
                    break

            self.w = np.sum(self.L * y_hat * x, axis=0)
            w_updates = np.abs(self.w - old_w)
            if np.max(w_updates) < tol:
                break

        b = y_hat - x.dot(self.w)
        if len(b[self.L.reshape(-1) > 0]) == 0:
            print("max iter")
        self.b = np.mean(b[self.L.reshape(-1) > 0])
        self._score = acc_v2(self.predict(x).reshape(-1), y.reshape(-1))
        if self._score <= .2:
            self.turn = -1
            self._score = 1 - self._score

    def _predict(self, x):
        pre = x.dot(self.w) + self.b
        return pre * self.turn

    def predict(self, x):
        pre = np.sign(self._predict(x))
        pre[pre == -1] = 0
        return pre

    def score(self):
        return self._score


class SVC(BaseSVM):
    def __init__(self):
        super(SVC, self).__init__()
        pass
