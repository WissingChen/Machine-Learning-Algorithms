# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : base.py
# @time    : 1/12/2021 上午10:17


class Base(object):
    def __init__(self):
        self._score = None
        pass

    def fit(self, x, y):
        """
        train the model
        :return:
        """
        pass

    def predict(self, x):
        """
        get the prediction from fitted model
        :return:
        """
        pass

    def score(self):
        """
        get score of the fitted model
        :return:
        """
        return self._score
