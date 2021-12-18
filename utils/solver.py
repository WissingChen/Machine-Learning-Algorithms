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
    def __init__(self, lr):
        self.lr = lr

    def step(self, layers, cache):
        cache.reverse()
        for i in range(len(layers)):
            dw, db = cache[i]
            layers[i].weight -= (self.lr * dw)
            layers[i].bias -= (self.lr * db)
        return layers
