# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _tree.py
# @time    : 1/12/2021 下午7:21
import numpy as np
from utils.metric.multi import acc_v2 as acc_v2_m
from utils.metric.binary import acc_v2
from utils.metric.regression import mse


class BaseTree(object):
    def __init__(self, criterion):
        self.criterion = None
        self.tree = None
        self._score = None
        self.get_criterion(criterion)

    def score(self):
        return self._score

    def get_criterion(self, criterion):
        self.criterion = criterion

    def fit(self, x, y):
        pass

    def _predict(self, x, value='value'):
        m = x.shape[0]
        _index = np.arange(m).reshape([-1, 1])
        x = np.concatenate([x, _index], axis=1)
        pre = self.tree.forward(x, value)
        return pre[pre[:, 0].argsort()][:, 1:]

    def predict(self, x, value):
        return self._predict(x, value)

    def make_tree(self, data, tree_type='classifier', n_classes=0, node=None, name='l'):
        """
        recursively spanning trees
        :param data:
        :param tree_type:
        :param n_classes: it can be any value in regression tree
        :param node: create leaf based on the node
        :param name: 'l' is left leaf, and 'r' is right leaf
        :return:
        """
        _index, node_value = self.criterion(data)
        a = np.argmin(_index)
        if node is None:
            self.tree = TreeNode([a, node_value[a]], data[:, -1], n_classes, _index=_index[a], _type=tree_type)
            data_a, data_b = self.split_data(data, [a, node_value[a]])
            self.make_tree(data_a, tree_type, n_classes, self.tree, "l")
            self.make_tree(data_b, tree_type, n_classes, self.tree, "r")
        else:
            leaf = TreeNode([a, node_value[a]], data[:, -1], n_classes, _index=_index[a], _type=tree_type)
            node.set_leaf(leaf, name)
            if len(np.unique(data[:, -1])) == 1:
                return None

            data_a, data_b = self.split_data(data, [a, node_value[a]])
            self.make_tree(data_a, tree_type, n_classes, leaf, "l")
            self.make_tree(data_b, tree_type, n_classes, leaf, "r")

    def split_data(self, data, node):
        """
        split data by node, yes mean <= the node value
        :param data:
        :param node: [which feature, value]
        :return: yes-data and no-data
        """
        node_name, value = node
        data_a = data[data[:, node_name] <= value]
        data_b = data[data[:, node_name] > value]
        return data_a, data_b


class DecisionTreeClassifier(BaseTree):
    """
    CART, multi classifier
    """

    def __init__(self, criterion='gini'):
        super(DecisionTreeClassifier, self).__init__(criterion)

    def get_criterion(self, criterion):
        if criterion == 'gini':
            self.criterion = self._gini_index

    def fit(self, x, y):
        data = np.concatenate([x, y.reshape([-1, 1])], axis=1)
        n_classes = len(np.unique(y.reshape(-1)))
        self.make_tree(data, 'classifier', n_classes)
        if n_classes == 2:
            self._score = acc_v2(self.predict(x, 'class').reshape(-1), y.reshape(-1))
        else:
            self._score = acc_v2_m(self.predict(x, 'class').reshape(-1), y.reshape(-1))

    def predict(self, x, value='class'):
        """
        :param x:
        :param value: which output value you want, choice [prob, class, value].
                    value is the sample value of in output leaf node.
        :return:
        """
        pre = self._predict(x, value)
        if value == 'class':
            return pre.reshape(-1)
        else:
            return pre

    """
        Criterion Function for leaf node split
    """

    @staticmethod
    def _gini(d):
        """
        Gini
        :param d: [n_sample, label]
        :return:
        """
        m = d.shape[0]
        Y = np.unique(d.reshape(-1))
        g = 0.
        for k in Y:
            g += ((np.sum(k == d) / m) ** 2)
        return 1 - g

    def _gini_index(self, d):
        """
        Gini index, continuous value
        :param d: [n_sample, n_feature + label]
        :return:
        """
        features = d[:, :-1]
        label = d[:, -1]
        m, n = features.shape
        g = np.ones([n])
        node = np.zeros([n])
        for i in range(n):
            V = np.unique(features[:, i]).reshape(-1)
            V = np.sort(V)
            T = [(V[i] + V[i + 1]) / 2.0 for i in range(len(V) - 1)]
            for v in T:
                a = features[:, i]
                mv = np.sum(a <= v)
                temp = (mv / m * self._gini(label[a <= v])) + ((1 - mv / m) * self._gini(label[a > v]))
                if g[i] > temp:
                    g[i] = temp
                    node[i] = v
        return g, node

    # todo correct the code of entropy
    @staticmethod
    def _entropy(d):
        """
        :param d: [n_sample, n_feature]
        :return:
        """
        m, n = d.shape
        g = np.zeros(n)
        for i in range(n):
            f = d[:, i]
            Y = np.unique(f).reshape(-1)
            for j in Y:
                g[i] -= ((np.sum(f == j) / m) * np.log((np.sum(f == j) / m)))
        return g


class DecisionTreeRegression(BaseTree):
    def __init__(self, criterion='mse'):
        super(DecisionTreeRegression, self).__init__(criterion)

    def get_criterion(self, criterion):
        if criterion == 'mse':
            self.criterion = self._mse_index

    def fit(self, x, y):
        data = np.concatenate([x, y.reshape([-1, 1])], axis=1)
        n_classes = 0
        self.make_tree(data, 'regression', n_classes)
        self._score = mse(self.predict(x).reshape(-1), y.reshape(-1))

    def predict(self, x, value='value'):
        """
        :param x:
        :param value: output the mean value of the sample values in leaf node
        :return:
        """
        pre = self._predict(x, value)
        return pre.reshape(-1)

    """
        Criterion Function for leaf node split
    """
    def _mse_index(self, d):
        """
        MSE index, continuous value
        :param d: [n_sample, n_feature + label]
        :return:
        """
        features = d[:, :-1]
        m, n = features.shape
        mse = np.ones([n]) * 1.e10
        node = np.zeros([n])
        for i in range(n):
            V = np.unique(features[:, i]).reshape(-1)
            V = np.sort(V)
            T = [(V[i] + V[i + 1]) / 2.0 for i in range(len(V) - 1)]
            for v in T:
                a = features[:, i]
                l_bag = d[a <= v]
                r_bag = d[a > v]
                temp = ((l_bag[:, -1].mean() - l_bag[:, -1]) ** 2).sum() / m + (
                            (r_bag[:, -1].mean() - r_bag[:, -1]) ** 2).sum() / m
                if mse[i] > temp:
                    mse[i] = temp
                    node[i] = v
        return mse, node


class TreeNode(object):
    def __init__(self, node, y, n_classes=0, _index=None, _type="classifier"):
        """
        create a node of the tree
        :param node: maybe like which feature and the split value, [feature index, (<=)split value]
        :param y: label of the sample in this node
        :param n_classes: the classes number of the whole raw data
        :param _index: the index of the node, such as gini index
        :param _type: classifier or regression
        """
        y = y.reshape(-1)
        feature_index, split_value = node
        self.node = feature_index
        self.split_value = split_value
        self.sample = y.shape[0]
        self._index = _index
        self._type = _type

        if self._type == "classifier":
            self._value = np.array([np.sum(y == i) for i in range(n_classes)]).reshape(-1)
            self._prob = self._value / self.sample
            self._class = np.argmax(self._value)
        elif self._type == "regression":
            self._value = y.mean()
        self.l_leaf = None
        self.r_leaf = None

    # def get_class(self, x):

    def set_leaf(self, leaf, name='l'):
        if name == 'l':
            self.l_leaf = leaf
        else:
            self.r_leaf = leaf

    def forward(self, x, value):
        if self.l_leaf is None or self.r_leaf is None:
            m = x.shape[0]
            if m == 0:
                return False
            if value == 'class':
                return np.concatenate([x[:, -1:].reshape([-1, 1]), np.repeat(self._class, m).reshape([m, -1])], axis=1)
            elif value == 'value':
                return np.concatenate(
                    [x[:, -1:].reshape([-1, 1]), np.repeat(self._value.reshape(1, -1), m, axis=0).reshape([m, -1])],
                    axis=1)
            elif value == 'prob':
                return np.concatenate(
                    [x[:, -1:].reshape([-1, 1]), np.repeat(self._prob.reshape(1, -1), m, axis=0).reshape([m, -1])],
                    axis=1)
        l_x = x[x[:, self.node] <= self.split_value]
        r_x = x[x[:, self.node] > self.split_value]
        l_pre = self.l_leaf.forward(l_x, value)
        r_pre = self.r_leaf.forward(r_x, value)
        if l_pre is not False and r_pre is not False:
            return np.concatenate([l_pre, self.r_leaf.forward(r_x, value)], axis=0)
        elif l_pre is False:
            return r_pre
        else:
            return l_pre
