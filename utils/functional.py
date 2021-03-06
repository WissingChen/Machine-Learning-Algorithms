# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : functional.py
# @time    : 13/12/2021 下午12:26
import numpy as np


def identity(x):
    """
    no-op activation, useful to implement linear bottleneck
    """
    return x


def sigmoid(x):
    """
    the logistic sigmoid function
    1 / (1 + np.exp(-x)) it will cause overflow in exp
    """
    return .5 * (1 + np.tanh(.5 * x))


def tanh(x):
    """
    the hyperbolic tan function
    """
    return np.tanh(x)


def relu(x):
    """
    the rectified linear unit function
    """
    return np.maximum(x, 0)


def leakyrelu(x):
    return np.maximum(x, x/10.)


def softmax(x, dim=1):
    _max = np.max(x, axis=dim, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - _max)  # subtracts each row with its max value
    _sum = np.sum(e_x, axis=dim, keepdims=True)  # returns sum of each row and keeps same dims
    y = e_x / _sum
    return y


class Identity(object):
    def __init__(self):
        self.x = None

    def __call__(self, x):
        self.x = x
        return x

    def backward(self, dy):
        return self.x


class Softmax(object):
    def __init__(self):
        self.x = None

    def __call__(self, x, dim=1):
        self.x = x
        y = softmax(x, dim)
        return y
    # todo grid
    # work with cross entropy loss


class Relu(object):
    def __init__(self):
        self.x = None

    def __call__(self, x):
        self.x = x
        return relu(x)

    def backward(self, dy):
        return np.multiply(dy, np.int64(self.x > 0))


class LeakyRelu(object):
    def __init__(self):
        self.x = None

    def __call__(self, x):
        self.x = x
        return leakyrelu(x)

    def backward(self, dy):
        grid = np.multiply(dy, np.int64(self.x > 0))
        return np.multiply(grid, .1 * np.float64(self.x <= 0))
