# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _svm.py
# @time    : 4/2/2022 下午8:11
import numpy as np
from utils.param import init_params
from utils.metric.binary import acc_v2
from utils.metric.regression import mse
from .kernel_func import RBFKernel, LinearKernel, PolynomialKernel, LaprasKernel, SigmoidKernel


class BaseSVM(object):
    """
    :param kernel: {'linear', 'poly', 'rbf', 'sigmoid', 'lapras'}, default='rbf'
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'lapras' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    :param degree: int, default=3
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    :param gamma: {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.

    :param C: float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        The penalty is a squared l2 penalty.
    """

    def __init__(self, C=1., kernel='rbf', degree=3., gamma='scale'):
        self.C = C
        self.w = None
        self.xi = None
        self.yi = None
        self.b = 0
        self.alpha = None
        self.degree = degree
        self.gamma = gamma

        self.kernel_fn = kernel
        self.kernel = None

        self._score = None

    def fit(self, x, y, tol=1.e-3):
        pass

    def get_gamma(self, x):
        _, n_features = x.shape
        return 1 / (n_features * x.var()) if self.gamma == 'scale' else 1 / n_features

    def get_kernel(self, x):
        if self.kernel_fn == 'linear':
            return LinearKernel()
        elif self.kernel_fn == 'rbf':
            return RBFKernel(self.get_gamma(x))
        elif self.kernel_fn == 'poly':
            return PolynomialKernel(self.degree, self.get_gamma(x))
        elif self.kernel_fn == 'lapras':
            return LaprasKernel(self.get_gamma(x))
        elif self.kernel_fn == 'sigmoid':
            return SigmoidKernel(self.get_gamma(x))

    def _predict(self, x):
        pre = np.sum(self.kernel(x, self.xi) * (self.alpha * self.yi).T, axis=1) + self.b
        # pre = x.dot(self.w) + self.b
        return pre

    def predict(self, x):
        return self._predict(x)

    def score(self):
        return self._score


class SVC(BaseSVM):
    def __init__(self, C=1., kernel='linear', degree=3., gamma='scale'):
        super(SVC, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma)
        self.turn = 1

    def fit(self, x, y, tol=1.e-3):
        m, n = x.shape
        self.xi = x.copy()
        self.yi = y.copy()
        self.yi[self.yi == 0] = -1
        self.kernel = self.get_kernel(self.xi)
        self.alpha = init_params(m, 1)

        for _ in range(m):
            old_a = self.alpha
            ai = np.argmax(self.alpha)
            aj = np.argmin(self.alpha)
            while True:
                old_ai = self.alpha[ai]
                a_y = np.sum(self.alpha * self.yi)
                a_i = self.alpha[ai] * self.yi[ai]
                a_j = self.alpha[aj] * self.yi[aj]
                c = -a_y + a_i + a_j
                self.alpha[ai] = (c - 2 / (self.yi[ai] * self.kernel(self.xi[ai], self.xi[aj]))) / (2 * self.yi[ai])
                # self.alpha[ai] = c / (2 * y_hat[ai] * y_hat[aj] * x[ai].dot(x[aj].T)) ????
                self.alpha[aj] = (c - self.alpha[ai] * self.yi[ai]) / self.yi[aj]
                if 0 <= self.alpha[ai] <= self.C or old_ai == self.alpha[ai]:
                    break

            a_updates = np.abs(self.alpha - old_a)
            if np.max(a_updates) < tol:
                break

        b = self.yi - np.sum(self.kernel(self.xi, self.xi) * self.alpha * self.yi, axis=1)
        self.b = np.mean(b[self.alpha.reshape(-1) > 0])
        self._score = acc_v2(self.predict(self.xi).reshape(-1), y.reshape(-1))
        if self._score <= .45:
            self.turn = -1
            self._score = 1 - self._score

    def _predict(self, x):
        pre = np.sum(self.kernel(x, self.xi) * (self.alpha * self.yi).T, axis=1) + self.b
        pre = np.sign(pre * self.turn)
        pre[pre == -1] = 0
        return pre


class SVR(BaseSVM):
    """
    :param C: float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
        The penalty is a squared l2 penalty.

    :param epsilon: float, default=0.1
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value.

    :param kernel: {'linear', 'poly', 'rbf', 'sigmoid', 'lapras'}, default='rbf'
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'lapras' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    :param degree: int, default=3
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    :param gamma: {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf'.

        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.

    """

    def __init__(self, C=1., epsilon=0.1, kernel='rbf', degree=3., gamma='scale'):
        super(SVR, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma)
        self.epsilon = epsilon

    def fit(self, x, y, tol=1.e-3):
        """
        fit Support Vector Regression
        :param x: [n_sample, n_feature]
        :param y: [n_sample, n_output]
        :param tol: float, default=1e-3
                    Tolerance for stopping criterion.
        :return:
        """
        m, n = x.shape
        self.xi = x.copy()
        self.yi = y.copy()
        self.kernel = self.get_kernel(x)
        self.alpha = init_params(m, 2)  # [a, a_hat]

        for _ in range(m):
            old_a = self.alpha
            ai = np.argmax(self.alpha[:, 0])
            aj = np.argmin(self.alpha[:, 0])

            # optimize alpha
            while True:
                old_ai = self.alpha[ai, 0]
                a_y = np.sum(self.alpha[:, 1] - self.alpha[:, 0])
                c = -a_y + (self.alpha[ai, 1] - self.alpha[ai, 0]) + (self.alpha[aj, 1] - self.alpha[aj, 0])
                self.alpha[ai: 0] = self.alpha[ai: 1] - c / 2. + (self.epsilon + self.yi[ai]) / self.kernel(self.xi[ai], self.xi[aj])
                self.alpha[aj: 0] = self.alpha[aj: 1] - c + self.alpha[ai: 1] - self.alpha[ai: 0]
                t_a = self.alpha[ai, 0]
                if 0 <= self.alpha[ai, 0] <= self.C or old_ai == self.alpha[ai, 0]:
                    break

            # optimize alpha_hat
            ai = np.argmax(self.alpha[:, 1])
            aj = np.argmin(self.alpha[:, 1])
            while True:
                old_ai = self.alpha[ai, 1]
                a_y = np.sum(self.alpha[:, 1] - self.alpha[:, 0])
                c = -a_y + (self.alpha[ai, 1] - self.alpha[ai, 0]) + (self.alpha[aj, 1] - self.alpha[aj, 0])
                self.alpha[ai: 1] = self.alpha[ai: 0] + c / 2. + (self.epsilon - self.yi[ai]) / self.kernel(self.xi[ai], self.xi[aj])
                self.alpha[aj: 1] = self.alpha[aj: 0] + c - self.alpha[ai: 1] + self.alpha[ai: 0]
                if 0 <= self.alpha[ai, 1] <= self.C or old_ai == self.alpha[ai, 1]:
                    break

            a_updates = np.abs(self.alpha - old_a)
            if np.max(a_updates) < tol:
                break

        b = self.yi + self.epsilon - np.sum(self.kernel(self.xi, self.xi) * (self.alpha[:, 1:] - self.alpha[:, :1]), axis=0)
        self.b = np.mean(b[self.alpha[:, 1] != self.alpha[:, 0]])
        self._score = mse(self.predict(x), y)

    def _predict(self, x):
        pre = np.sum(self.kernel(x, self.xi) * (self.alpha[:, 1:] - self.alpha[:, :1]).T, axis=1) + self.b
        return pre
