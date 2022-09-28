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
from utils.metric.multi import acc_v2 as acc_v2_m
from utils.metric.binary import acc_v2
from utils.metric.regression import mse
from utils.param import init_params, one_hot

n_activation = {"relu": _f.Relu, "tanh": _f.tanh, "sigmoid": _f.sigmoid, 'none': _f.Identity}
n_solver = {"sgd": _solver.SGD}

# todo l1 and l2


class MLPBase(object):
    """
    :param dim: hidden dim
    """
    def __init__(self, dim=(100,), activation='relu', solver="sgd", alpha=0.0001,
                 learning_rate=1e-3, val_ratio=0.1):
        super(MLPBase, self).__init__()
        self.dim = dim
        self.network = None
        self.activation = [n_activation[activation]() for _ in range(len(dim))]
        self.val_ratio = val_ratio

        self.solver = n_solver[solver](learning_rate)
        self.cache = []
        self.loss_fn = None
        self._score = None

    def zero_grid(self):
        del self.cache
        self.cache = []

    def preprocess_data(self, y):
        return y

    def fit(self, x, y, epoch=100, batch_size=200, log=False):
        m, n = x.shape
        val_m = int(self.val_ratio * m)
        train_m = m - val_m
        y_hat = self.preprocess_data(y)
        dim = (n,) + self.dim + (y_hat.shape[1],)
        self.network = [Layer(in_ch=dim[i], out_ch=dim[i + 1]) for i in range(len(dim) - 1)]

        data = np.concatenate([x, y_hat.reshape([m, -1])], axis=1)
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

                sample_y = train_data[:, n:]
                pre = self.forward(train_data[:, :n])
                self.zero_grid()
                loss = self.backward(pre, sample_y)
                self.solver.step(self.network, self.cache)
                if log:
                    print(f"{ep + 1}/{epoch}\tTrain:\tLoss={loss:.4f}")
            # val
            for _index in range(0, val_m, batch_size):
                if _index + batch_size < val_m:
                    val_data = val_set[_index: _index + batch_size]
                else:
                    val_data = val_set[_index:]
                pre = self.forward(val_data[:, :n])
                sample_y = val_data[:, n:]
                loss = self.backward(pre, sample_y)
                # print(self.network[0].weight[0, :10])
                if log:
                    print(f"{ep + 1}/{epoch}\tVal:\tLoss={loss:.4f}")

        self.score_fn(x, y)

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

    def _predict(self, x):
        return self.forward(x)

    def predict(self, x):
        return self._predict(x)

    def score(self):
        return self._score

    def score_fn(self, x, y):
        self._score = None


class MLPClassifier(MLPBase):
    """
    Default to use the validation set and early stopCross entropyThe final output is using softmax
    """

    def __init__(self, dim=(100,), activation='relu', solver="sgd", alpha=0.0001,
                 learning_rate=1.e-3, val_ratio=0.1):
        super(MLPClassifier, self).__init__(dim, activation, solver, alpha, learning_rate, val_ratio)
        self.loss_fn = _loss.CrossEntropyLoss()

    def preprocess_data(self, y):
        return one_hot(y)

    def forward(self, x):
        for i in range(len(self.network) - 1):
            layer = self.network[i]
            h = layer.forward(x)
            x = self.activation[i](h)
        layer = self.network[-1]
        h1 = layer.forward(x)
        y = _f.softmax(h1)
        return y

    def score_fn(self, x, y):
        pre = self.predict(x)
        n_classes = np.max(y) + 1
        y = y.reshape(-1)
        if n_classes == 2:
            self._score = acc_v2(pre, y)
        else:
            self._score = acc_v2_m(pre, y)

    def predict(self, x, value='class'):
        pre = self._predict(x)
        if value == 'class':
            return np.argmax(pre, axis=1)
        elif value == 'prob':
            return pre


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

    def score_fn(self, x, y):
        pre = self.forward(x)
        self._score = mse(pre, y)


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
