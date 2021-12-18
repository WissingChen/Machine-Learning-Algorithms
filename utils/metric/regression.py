# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : regression.py
# @time    : 30/11/2021 下午8:53
import numpy as np


def mse(pre, y):
    """
    MSE - Mean Squared Error
    """
    return np.mean((pre - y)**2)


def mae(pre, y):
    """
    MAE - Mean Absolute Error
    """
    return np.mean(np.abs(pre - y))
