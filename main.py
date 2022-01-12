# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : main.py
# @time    : 2021/7/21 13:06

from sklearn.datasets import load_breast_cancer  # 2 classes
from sklearn.datasets import load_iris  # 3 classes
from sklearn.datasets import load_digits  # image   10 classes
from sklearn.datasets import load_diabetes  # regression

from utils.metric.regression import mse, spearman_corr
from utils.metric.binary import accuracy, auc, roc
from utils.metric.multi import accuracy as _accuracy

import numpy as np
from model.linear import Lasso as ME
from sklearn.linear_model import Lasso as SK

dataset = {'regression': load_diabetes, 'binary': load_breast_cancer,
           'multi': load_iris, 'cv': load_digits}

task = ['regression', 'binary', 'multi', 'cv']

if __name__ == '__main__':
    task_id = 0
    x, y = dataset[task[task_id]](return_X_y=True)
    m, n = x.shape
    data = np.concatenate([x, y.reshape([m, -1])], axis=1)
    np.random.shuffle(data)
    _m = int(data.shape[0] / 10 * 7)
    train = data[_m:]
    test = data[:_m]
    # model
    model = ME()
    model_gt = SK()
    # train
    model.fit(train[:, :n], train[:, n:])
    model_gt.fit(train[:, :n], train[:, n:])
    # test
    pre = model.predict(test[:, :n])
    pre_gt = model_gt.predict(test[:, :n])
    # score
    print(model.score())
    if task_id == 0:
        print("my model: ", mse(pre, test[:, n:]),
              "\nsci-kit model: ", mse(pre_gt, test[:, n:]))
    elif task_id == 1:
        print("my model: ", accuracy(pre[: 1], test[:, n]),
              "\nsci-kit model: ", accuracy(pre_gt[: 1], test[:, n]))
    elif task_id == 2:
        print("my model: ", _accuracy(pre, test[:, n]),
              "\nsci-kit model: ", _accuracy(pre_gt, test[:, n]))
    elif task_id == 3:
        print("my model: ", accuracy(pre, test[:, n:]),
              "\nsci-kit model: ", accuracy(pre_gt, test[:, n:]))
