# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _bayes.py
# @time    : 15/2/2022 下午3:46
import numpy as np


class BaseNB(object):
    def __init__(self):
        self.data = None
        self._score = None
        self.p_label = {}
        self.p_feature = {}
        self.p_feature_label = {}

    def score(self):
        return self._score

    def fit(self, x, y):
        n_classes = int(np.max(y) + 1)
        n_sample, n_feature = x.shape
        self.data = np.concatenate([x, y.reshape([n_sample, -1])], axis=1)
        self.p_label = np.zeros([n_classes])
        # P(Y) key -> label
        for i in range(n_classes):
            self.p_label[i] = np.sum(y == i) / n_sample
        # P(X) key -> {feature id}_{feature value}
        for i in range(n_feature):
            x_hat = np.unique(x[:, i])
            for j in range(len(x_hat)):
                self.p_feature[f"{i}_{x_hat[j]}"] = np.sum(x[:, i] == x_hat[j]) / n_sample
        # P(X|Y) key -> {label}_{feature id}_{feature value}
        for i in range(n_classes):
            x_hat = x[y.reshape(-1) == i]
            for j in range(n_feature):
                _x_hat = np.unique(x_hat[:, j])
                for k in range(len(_x_hat)):
                    self.p_feature_label[f"{i}_{j}_{_x_hat[k]}"] = np.sum(x_hat[:, j] == _x_hat[k]) / n_sample

    def _predict(self, x):
        # TODO Needs to be optimized
        n = len(self.p_label)
        pre = np.zeros([x.shape[0], n])
        for i in range(x.shape[0]):
            pass
        return pre

    def predict(self, x):
        return self._predict(x)

