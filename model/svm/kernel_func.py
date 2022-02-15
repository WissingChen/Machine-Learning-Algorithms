# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : kernel_func.py
# @time    : 14/2/2022 下午11:39
import numpy as np

"""
Kernel function return data with shape [n_sample1, n_sample_2]
"""


class LinearKernel(object):
    def __init__(self):
        pass

    def __call__(self, xi, xj):
        return xi.dot(xj.T)


class PolynomialKernel(object):
    def __init__(self, degree, gamma):
        self.degree = degree
        self.gamma = gamma

    def __call__(self, xi, xj):
        return np.power((self.gamma * xi.dot(xj.T)), self.degree)


class RBFKernel(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, xi, xj):
        if len(xi.shape) == 1:
            dst = np.exp(-self.gamma * np.sum((xi - xj) ** 2))
            return dst
        else:
            mi = xi.shape[0]
            mj = xj.shape[0]
            dst = np.zeros([mi, mj])
            for i in range(mi):
                dst[i] = np.exp(-self.gamma * np.sum((xi[i] - xj) ** 2, axis=1).T)
            return dst


class LaprasKernel(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, xi, xj):
        return np.exp(-(np.abs(xi - xj)) / self.gamma)


class SigmoidKernel(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, xi, xj):
        return np.tanh(xi.dot(xj.T) * self.gamma)
