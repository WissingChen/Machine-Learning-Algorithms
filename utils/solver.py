# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : solver.py
# @time    : 1/12/2021 下午1:19
import numpy as np


class SGD(object):
    """
    for neural network
    """
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def step(self, layers, cache):
        cache.reverse()
        for i in range(len(layers)):
            dw, db = cache[i]
            layers[i].velocity_w = self.lr * dw + self.momentum * layers[i].velocity_w
            layers[i].velocity_b = self.lr * db + self.momentum * layers[i].velocity_b
            layers[i].weight -= layers[i].velocity_w
            layers[i].bias -= layers[i].velocity_b
        return layers


class Adam(object):
    def __init__(self):
        pass


class AdaGrad(object):
    def __init__(self):
        pass
