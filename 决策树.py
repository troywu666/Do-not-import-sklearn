'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-06-30 09:12:32
@LastEditors: Troy Wu
@LastEditTime: 2020-07-01 15:53:18
'''
import numpy as np

class DecisionTree:
    '''ID3算法'''
    class Node:
        def __init__(self):
            self.value = None
            # 内部叶子节点属性
            self.feature_index = None
            self.children = {}
            
        def __str__(self):
            if self.children:
                s = '内部节点<{}>:\n'.format(self.feature_index)
                for fv, node in self.children.items():
                    ss = '[{}]->{}'.format(fv, node)
                    s += '\t' + ss.replace('\n', '\n\t') + '\n'
            else:
                s = '叶节点({})'.format(self.value)
            return s
    
    def __init__(self, gain_threshold = 1e-2):
        # 信息增益阈值
        self.gain_threshold = gain_threshold
        
    def _entropy(self, y):
        '''熵'''
        c = np.bincount(y)
        p = c[np.nonzero(c)] / y.size
        return np.sum(p * np.log2(p)) * -1.0
    
    def _conditional_entropy(self, feature, y):
        '''条件熵'''
        feature_values = np.unique(feature)
        h = 0.
        for v in feature_values:
            y_sub = y[feature == v]
            p = y_sub.size / y.size
            h += p * self._entropy(y_sub)
        return h

    def _information_gain(self, feature, y):
        return self._entropy(y) - self._conditional_entropy(feature, y)

    def _select_feature(self, X, y, feature_list):
        '''选择信息增益最大的特征'''
        if feature_list:
            gains = np.apply_along_axis(self._information_gain, 0, X[:, feature_list], y)
            index = np.argmax(gains)
            if gains[index] > self.gain_threshold:
                return index
        # 当feature_list已为空，或所有特征信息增益都小于阈值时，返回None
        return None

    def _build_tree(self, X, y, feature_list):
        # 创造节点
        node = DecisionTree.Node()
        # 统计数据集中样本类标记的个数
        labels_count = np.bincount(y)
        # 任何情况下节点值总等于数据集中样本最多的类标记
        node.value = np.argmax(np.bincount(y))

        if np.count_nonzero(labels_count) != 1:
            # 选择信息增益最大的特征
            index = self._select_feature(X, y, feature_list)
            # 能选择到适合的特征时， 创建内部节点，否则创建叶节点
            if index is not None:
                node.feature_index = feature_list.pop(index)
                feature_values = np.unique(X[:, node.feature_index])
                for v in feature_values:
                    idx = X[:, node.feature_index] == v
                    X_sub, y_sub = X[idx], y[idx]
                    # 创建子树
                    node.children[v] = self._build_tree(X_sub, y_sub, feature_list.copy())
        return node

    def _predict_one(self, x):
        '''搜索决策树，对单个实例进行预测'''
        node = self.tree_
        while node.children:
            child = node.children.get(x[node.feature_index])
            if not child:
                break
            node = child
        return node.value

    def train(self, X_train, y_train):
        _, n = X_train.shape
        self.tree_ = self._build_tree(X_train, y_train, list(range(n)))
        
    def predict(self, X_test):
        return np.apply_along_axis(self._predict_one, axis = 1, arr = X_test)
        
    def __str__(self):
        if hasattr(self, 'tree_'):
            return str(self.tree_)
        return ''

class CartClassificationTree:
    class Node:
        def __init__(self):
            self.value = None
            self.feature_index = None
            self.feature_value = None
            self.left = None
            self.right = None

        def __str__(self):
            if self.left:
                s = '内部节点<{}>\n'.format(self.feature_index)
                ss = '[>{}]->{}'.format(self.feature_index, self.left)
                s += '\t' + ss.replace('\n', '\n\t') + '\n'
                ss = '[<={}]->{}'.format(self.feature_value, self.right)
                s += '\t' + ss.replace('\n', '\n\t')
            else:
                s = '叶节点({})'.format(self.value)
            return s

    def __init__(self, gini_threshold = 0.01, gini_dec_threshold = 0., min_samples_split = 2):
        # 基尼系数降低的阈值
        self.gini_dec_threshold = gini_dec_threshold
        # 基尼系数的阈值
        self.gini_threshold = gini_threshold
        # 数据集还可继续切分的最小样本数量
        self.min_samples_split = min_samples_split
    
    def _gini(self, y):
        values = np.unique(y)
        s = 0.
        for v in values:
            y_sub = y[y == v]
            s += (y_sub.size / y.size) ** 2
        return 1 - s

    def _gini_split(self, y, feature, value):
        '''计算根据特征切分后的基尼系数'''
        indices = feature > value
        y1 = y[indices]
        y2 = y[~indices]
        gini1 = self._gini(y1)
        gini2 = self._gini(y2)
        gini = (y1.size * gini1 + y2.size * gini2) / y.size
        return gini

    def _get_split_points(self, feature):
        '''获得一个连续特征的所有切分点'''
        values = np.unique(feature)
        split_points = [(v1+v2)/2 for v1, v2, in zip(values[: -1], values[1:])]
        return split_points

    def _select_feature(self, X, y):
        '''选择划分特征'''
        best_feature_index = None
        best_split_value = None
        min_gini = np.inf
        _, n = X.shape
        for feature_index in range(n):
            feature = X[:, feature_index]
            split_points = self._get_split_points(feature)
            for value in split_points:
                # 迭代每一个分割点value，计算使用value分割后的数据集基尼系数
                gini = self._gini_split(y, feature, value)
                if gini < min_gini:
                    min_gini = gini
                    best_feature_index = feature_index
                    best_split_value = value

        if self._gini(y) - min_gini < self.gini_dec_threshold:
            best_feature_index = None
            best_split_value = None
        return best_feature_index, best_split_value, min_gini

    def _node_value(self, y):
        '''计算节点的值'''
        label_counts = np.bincount(y)
        return np.argmax(label_counts)

    def _create_tree(self, X, y):
        '''生成树递归算法'''
        # 创建节点
        node = self.Node()
        # 计算节点的值
        node.value = self._node_value(y)
        # 若当前数据集样本数量小于最小分割数量min_samples_split，则返回叶节点
        if y.size < self.min_samples_split:
            return node
        # 若当前数据集的基尼系数小于阈值gini_threshold，则返回叶节点
        if self._gini(y) < self.gini_threshold:
            return node

        # 选择最佳分割特征
        feature_index, feature_value, min_gini = self._select_feature(X, y)
        if feature_index is not None:
            node.feature_index = feature_index
            node.feature_value = feature_value
            feature = X[:, feature_index]
            indices = feature > feature_value
            X1, y1 = X[indices], y[indices]
            X2, y2 = X[~indices], y[~indices]
            node.left = self._create_tree(X1, y1)
            node.right = self._create_tree(X2, y2)

        return node

    def _predict_one(self, X_test):
        node = self.tree_
        while node.left:
            if X_test[node.feature_index] < node.feature_value:
                node = node.left
            else:
                node = node.right
        return node.value

    def train(self, X_train, y_train):
        self.tree_ = self._create_tree(X_train, y_train)

    def predict(self, X_test):
        return np.apply_along_axis(self._predict_one, axis = 1, arr = X_test)

class CartRegressionTree:
    class Node:
        def __init__(self):
            self.value = None
            self.feature_index = None
            self.feature_value = None
            self.left = None
            self.right = None
        
        def __str__(self):
            if self.children:
                s = '内部节点<%s>:\n' % self.feature_index
                for fv, node in self.children.items():
                    ss = '[%s]-> %s' % (fv, node)
                    s += '\t' + ss.replace('\n', '\n\t') + '\n'
                s = s[:-1]
            else:
                s = '叶节点(%s)' % self.value
            return s

    def __init__(self, mse_threshold = 0.01, mse_dec_threshold = 0., min_samples_split = 2):
        self.mse_threshold = mse_threshold
        self.mse_dec_threshold = mse_dec_threshold
        self.min_samples_split = min_samples_split
    
    def _mse(self, y):
        return np.var(y)

    def _mse_split(self, y, feature, value):
        indices = feature > value
        y1 = y[indices]
        y2 = y[~indices]
        mse1 = self._mse(y1)
        mse2 = self._mse(y2)
        return (y1.size * mse1 + y2.size * mse2) / y.size

    def _get_split_points(self, feature):
        values = np.unique(feature)
        split_points = [(v1+v2)/2 for v1, v2 in zip(values[:-1], values[1:])]
        return split_points

    def _select_feature(self, X, y):
        # 最佳分割特征的index
        best_feature_index = None
        # 最佳分割点
        best_split_value = None
        min_mse = np.inf
        _, n = X.shape
        for feature_index in range(n):
            feature = X[:, feature_index]
            split_points = self._get_split_points(feature)
            for value in split_points:
                # 迭代每一个分割点value，计算使用value分割后的数据集mse
                mse = self._mse_split(y, feature, value)
                if mse < min_mse:
                    min_mse = mse
                    best_feature_index = feature_index
                    best_split_value = value
        if self._mse(y) - min_mse < self.mse_dec_threshold:
            best_feature_index = None
            best_split_value = None

        return best_feature_index, best_split_value, min_mse

    def _node_value(self, y):
        '''计算节点的值'''
        return np.mean(y)

    def _create_tree(self, X, y):
        node = self.Node()
        node.value = self._node_value(y)
        if y.size < self.min_samples_split:
            return node
        if self._mse(y) < self.mse_threshold:
            return node

        # 选择最佳分割特征
        feature_index, feature_value, min_mse = self._select_feature(X, y)
        if feature_index is not None:
            node.feature_index = feature_index
            node.feature_value = feature_value
            feature = X[:, feature_index]
            indices = feature < feature_value
            X1, y1 = y[indices]
            X2, y2 = y[~indices]
            node.left = self._create_tree(X1, y1)
            node.right = self._create_tree(X2, y2)
        return node

    def _predict_one(self, X_test):
        node = self.tree_
        while node.left:
            if X_test[node.feature_index] > node.feature_value:
                node = node.left
            else:
                node.right
        return node.value

    def train(self, X_train, y_train):
        self.tree_ = self._create_tree(X_train, y_train)
        
    def predict(self, X_test):
        return np.apply_along_axis(self._predict_one, axis = 1, arr = X_test)