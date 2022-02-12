# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : temp.py
# @time    : 11/12/2021 上午1:07
import numpy as np
from utils.param import one_hot

m = 100
n_classes = 2
y = np.zeros([100, 1]).astype(np.int8)
y[::2, 0] = 1
print(np.mean(y))
