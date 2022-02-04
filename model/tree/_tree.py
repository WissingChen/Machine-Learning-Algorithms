# @encoding: utf-8
# @author  : wissingcc
# @contact : chen867820261@gmail.com
# @software: PyCharm
# @file    : _tree.py
# @time    : 1/12/2021 下午7:21
import numpy as np
from utils.metric.multi import acc_v2 as acc_v2_m
from utils.metric.binary import acc_v2


class BaseTree(object):
    def __init__(self, criterion):
        self.criterion = None
        self.tree = None
        self._score = None
        self.get_criterion(criterion)
        self._score_fn = None

    def score(self):
        return self._score

    def get_criterion(self, criterion):
        self.criterion = criterion

    def fit(self, x, y):
        pass

    def predict(self, x):
        m = x.shape[0]
        _index = np.arange(m).reshape([-1, 1])
        x = np.concatenate([x, _index], axis=1)
        pre = self.tree.forward(x)
        return pre[pre[:, 0].argsort()][:, 1]

    def make_tree(self, data, n_classes, node=None, name='l'):
        """
        recursively spanning trees
        :param data:
        :param n_classes:
        :param node: create leaf based on the node
        :param name: 'l' is left leaf, and 'r' is right leaf
        :return:
        """
        _index, node_value = self.criterion(data)
        a = np.argmin(_index)
        if node is None:
            self.tree = TreeNode([a, node_value[a]], data[:, -1], n_classes, _index=_index[a])
            data_a, data_b = self.split_data(data, [a, node_value[a]])
            self.make_tree(data_a, n_classes, self.tree, "l")
            self.make_tree(data_b, n_classes, self.tree, "r")
        else:
            leaf = TreeNode([a, node_value[a]], data[:, -1], n_classes, _index=_index[a])
            node.set_leaf(leaf, name)
            if len(np.unique(data[:, -1])) == 1:
                return None

            data_a, data_b = self.split_data(data, [a, node_value[a]])
            self.make_tree(data_a, n_classes, leaf, "l")
            self.make_tree(data_b, n_classes, leaf, "r")

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
        self.make_tree(data, n_classes)
        if n_classes == 2:
            self._score = acc_v2(self.predict(x).reshape(-1), y.reshape(-1))
        else:
            self._score = acc_v2_m(self.predict(x).reshape(-1), y.reshape(-1))

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


class DecisionTreeRegression(DecisionTreeClassifier):
    def __init__(self, criterion='mse'):
        super(DecisionTreeRegression, self).__init__()
        self.tree = None

        if criterion == 'mse':
            self.criterion = _gini_index

    """
        Criterion Function for leaf node split
    """
    def _mse(self):


class TreeNode(object):
    def __init__(self, node, y, n_classes, _index=None):
        """
        create a node of the tree
        :param node: maybe like which feature and the split value, [feature index, (<=)split value]
        :param y: label of the sample in this node
        :param n_classes: the classes number of the whole raw data
        :param _index: the index of the node, such as gini index
        """
        y = y.reshape(-1)
        feature_index, split_value = node
        self.node = feature_index
        self.split_value = split_value
        self.sample = y.shape
        self.value = [np.sum(y == i) for i in range(n_classes)]

        self._index = _index
        self._class = np.argmax(self.value)
        self.l_leaf = None
        self.r_leaf = None

    # def get_class(self, x):

    def set_leaf(self, leaf, name='l'):
        if name == 'l':
            self.l_leaf = leaf
        else:
            self.r_leaf = leaf

    def forward(self, x):
        if self.l_leaf is None or self.r_leaf is None:
            m = x.shape[0]
            return np.concatenate([x[:, -1:].reshape([-1, 1]), np.repeat(self._class, m).reshape([-1, 1])], axis=1)
        l_x = x[x[:, self.node] <= self.split_value]
        r_x = x[x[:, self.node] > self.split_value]
        pre = self.l_leaf.forward(l_x)
        pre = np.concatenate([pre, self.r_leaf.forward(r_x)], axis=0)
        return pre
