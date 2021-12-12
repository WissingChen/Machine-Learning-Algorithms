# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : temp.py
# @time    : 11/12/2021 上午1:07
import numpy as np


a = np.arange(7)
b = np.concatenate([np.zeros(3), np.ones(4)])
np.random.shuffle(a)
t = np.concatenate([a.reshape([-1, 1]), b.reshape([-1, 1])], axis=1)
print(t)
t = t[t[:, 0].argsort()]
print(t)

