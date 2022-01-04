# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : main.py
# @time    : 2021/7/21 13:06

from sklearn.datasets import load_breast_cancer  # 2 classes
# from sklearn.datasets import load_iris  # 3 classes
# from sklearn.datasets import load_digits  # image   10 classes
from sklearn.datasets import load_diabetes  # regression
from utils.metric.regression import mse
from utils.metric.binary import accuracy

import numpy as np
from model.linear import Ridge as ME
from sklearn.linear_model import Ridge as SK

if __name__ == '__main__':
    x, y = load_breast_cancer(return_X_y=True)
    m, n = x.shape
    data = np.concatenate([x, y.reshape([m, -1])], axis=1)
    np.random.shuffle(data)
    _m = int(data.shape[0] / 10 * 7)
    train = data[_m:]
    test = data[:_m]
    # model
    model = ME(solver='lsqr')
    model_gt = SK()
    model.fit(train[:, :n], train[:, n:])
    model_gt.fit(train[:, :n], train[:, n:])

    pre = model.predict(test[:, :n])
    pre_gt = model_gt.predict(test[:, :n])
    print(model.score())
    print("my model: ", accuracy(pre, test[:, n:]),
          "\nsci-kit model: ", accuracy(pre_gt, test[:, n:]))
