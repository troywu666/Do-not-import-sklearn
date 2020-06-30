'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-06-30 09:12:32
@LastEditors: Troy Wu
@LastEditTime: 2020-06-30 21:58:26
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