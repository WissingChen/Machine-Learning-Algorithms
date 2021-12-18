# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : main.py
# @time    : 2021/7/21 13:06
import numpy as np
from sklearn.datasets import load_breast_cancer  # 2 classes
from utils.metric.binary import accuracy
from model.nn import MLPClassifier as ME
from sklearn.neural_network import MLPClassifier as SK

if __name__ == '__main__':
    x, y = load_breast_cancer(return_X_y=True)
    # x = np.random.randn(500, 30)
    # y = np.random.randn(500, 1)
    data = np.concatenate([x, y.reshape([-1, 1])], axis=1)
    np.random.shuffle(data)
    m = int(data.shape[0] / 10 * 7)
    train = data[m:]
    test = data[:m]
    # model
    model = ME((30, 1000, 1000, 2))
    model_gt = SK(solver='sgd')
    model.fit(train[:, :-1], train[:, -1], 100)
    model_gt.fit(train[:, :-1], train[:, -1])

    pre = model.predict(test[:, :-1])
    pre_gt = model_gt.predict(test[:, :-1])
    print(model.score())
    print("my model: ", accuracy(pre[:, 1], test[:, -1]),
          "\nsci-kit model: ", accuracy(pre_gt, test[:, -1]))
