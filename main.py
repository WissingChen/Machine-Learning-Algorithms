# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : main.py
# @time    : 2021/7/21 13:06
import sklearn as sk
import numpy as np
from sklearn.datasets import load_iris  # 3 classes
from sklearn.datasets import load_digits  # image   10 classes
from sklearn.datasets import load_breast_cancer  # 2 classes
from sklearn.datasets import load_diabetes  # regression
from utils.metric_binary import accuracy
from utils.metric_regression import mse
from model.tree import DecisionTreeClassifier as ME
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier


if __name__ == '__main__':
    x, y = load_breast_cancer(return_X_y=True)
    data = np.concatenate([x, y.reshape([-1, 1])], axis=1)
    np.random.shuffle(data)
    m = int(data.shape[0]/10*3)
    train = data[m:]
    test = data[:m]

    model = ME()
    model_gt = DecisionTreeClassifier()
    model.fit(train[:, :-1], train[:, -1])
    model_gt.fit(train[:, :-1], train[:, -1])

    pre = model.predict(test[:, :-1])
    pre_gt = model_gt.predict(test[:, :-1])
    print(model.score())
    print(accuracy(pre, test[:, -1]), accuracy(pre_gt, test[:, -1]))
