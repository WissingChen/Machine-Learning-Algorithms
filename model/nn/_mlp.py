# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _mlp.py
# @time    : 11/12/2021 下午3:22
import numpy as np
import utils.functional as _f
from utils import solver as _solver
from utils import loss as _loss
from utils.metric.binary import roc, auc
from utils.param import init_params

n_activation = {"relu": _f.Relu, "tanh": _f.tanh, "sigmoid": _f.sigmoid, 'none': _f.Identity}
n_solver = {"sgd": _solver.SGD}


class MLPBase(object):
    def __init__(self, dim=(100,), activation='relu', solver="sgd", alpha=0.0001,
                 learning_rate=1e-3, val_ratio=0.1):
        super(MLPBase, self).__init__()
        self.network = [Layer(in_ch=dim[i], out_ch=dim[i + 1]) for i in range(len(dim) - 1)]
        self.activation = [n_activation[activation]() for _ in range(len(self.network) - 1)]
        self.val_ratio = val_ratio

        self.solver = n_solver[solver](learning_rate)
        self.cache = []
        self.loss_fn = None

    def zero_grid(self):
        del self.cache
        self.cache = []

    def fit(self, x, y, epoch=100, batch_size=200):
        m, n = x.shape
        val_m = int(self.val_ratio * m)
        train_m = m - val_m

        data = np.concatenate([x, y.reshape([m, -1])], axis=1)
        np.random.shuffle(data)
        train_set = data[val_m:]
        val_set = data[:val_m]

        for ep in range(epoch):
            np.random.shuffle(train_set)
            # train
            for _index in range(0, train_m, batch_size):
                if _index + batch_size < train_m:
                    train_data = train_set[_index: _index + batch_size]
                else:
                    train_data = train_set[_index:]
                y = train_data[:, -1:]
                pre = self.forward(train_data[:, :-1])
                self.zero_grid()
                loss = self.backward(pre, y)
                self.solver.step(self.network, self.cache)
                print(f"{ep + 1}/{epoch}\tTrain: Loss={loss:.4f}")
            # val
            for _index in range(0, val_m, batch_size):
                if _index + batch_size < val_m:
                    val_data = val_set[_index: _index + batch_size]
                else:
                    val_data = val_set[_index:]
                pre = self.forward(val_data[:, :-1])
                y = val_data[:, -1:]
                loss = self.backward(pre, y)
                # print(self.network[0].weight[0, :10])
                print(f"{ep + 1}/{epoch}\tVal: Loss={loss:.4f}")

    def backward(self, pre, y):
        loss = self.loss_fn(pre, y)
        dy = self.loss_fn.backward()
        dx, dw, db = self.network[-1].backward(dy)
        self.cache += [[dw, db]]
        for i in range(2, len(self.network) + 1):
            dy = self.activation[-i + 1].backward(dx)
            dx, dw, db = self.network[-i].backward(dy)
            self.cache += [[dw, db]]
        return loss

    def forward(self, x):
        pass

    def predict(self, x):
        return self.forward(x)


class MLPClassifier(MLPBase):
    """
    Default to use the validation set and early stopCross entropyThe final output is using softmax
    """

    def __init__(self, dim=(100, 100), activation='relu', solver="sgd", alpha=0.0001,
                 learning_rate=1.e-3, val_ratio=0.1):
        super(MLPClassifier, self).__init__(dim, activation, solver, alpha, learning_rate, val_ratio)
        self.loss_fn = _loss.CrossEntropyLoss()

    @staticmethod
    def _score(pre, y):
        x, y = roc(pre, y)
        return auc(x, y)

    def forward(self, x):
        for i in range(len(self.network) - 1):
            layer = self.network[i]
            h = layer.forward(x)
            x = self.activation[i](h)
        layer = self.network[-1]
        h1 = layer.forward(x)
        y = _f.softmax(h1)
        return y


class MLPRegression(MLPBase):
    def __init__(self, dim=(100,), activation='relu', solver="sgd", alpha=0.0001,
                 learning_rate=1e-3, val_ratio=0.1):
        super(MLPRegression, self).__init__(dim, activation, solver, alpha, learning_rate, val_ratio)
        self.loss_fn = _loss.MSELoss()

    def forward(self, x):
        for i in range(len(self.network) - 1):
            layer = self.network[i]
            h = layer.forward(x)
            x = self.activation[i](h)
        layer = self.network[-1]
        h1 = layer.forward(x)
        y = _f.identity(h1)
        return y


class Layer(object):
    def __init__(self, in_ch, out_ch):
        self.weight = init_params(in_ch, out_ch)
        self.bias = np.zeros([1, out_ch])
        self.h = None
        self.velocity_w = 0
        self.velocity_b = 0

    def forward(self, x):
        self.h = x.copy()
        y = np.dot(x, self.weight) + self.bias
        return y

    def backward(self, dy):
        dw = np.dot(self.h.T, dy)
        db = np.sum(dy, axis=0, keepdims=True)
        if dy.shape[1] == self.weight.shape[1]:
            dx = np.dot(dy, self.weight.T)
            return dx, dw, db
        return 0, dw, db
