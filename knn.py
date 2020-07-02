'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-02 15:50:56
@LastEditors: Troy Wu
@LastEditTime: 2020-07-02 22:11:47
'''
import numpy as np
from queue import deque
import heapq

class KNN:
    def __init__(self, k_neighbors = 5):
        self.k_neighbors = k_neighbors
        
    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _predict_one(self, x):
        d = np.linalg.norm(x - self.X_train, axis = 1)
        idx = np.argpartition(d, self.k_neighbors)[: self.k_neighbors]
        # 根据索引得到每个测试样本k_neighbors个邻居的y值
        y_neighbors = self.y_train[idx]
        counts = np.bincount(y_neighbors)
        return np.argmax(counts)
        
    def predict(self, X):
        return np.apply_along_axis(self._predict_one, axis = 1, arr = X)

class KDTree:
    def __init__(self, k_neighbors = 5):
        self.k_neighbors = k_neighbors

    def _node_depth(self, i):
        '''计算节点深度'''
        t = np.log2(i + 2)
        return int(t) + (0 if t.is_integer() else 1)

    def _kd_tree_build(self, X):
        m, n = X.shape
        tree_depth = self._node_depth(m - 1)
        # 节点由两个索引组成，[0]实例索引，[1]切分特征索引
        M = 2 ** tree_depth - 1
        tree = np.zeros((M, 2), dtype = int)
        tree[:, 0] = -1
        # 使用队列桉树的层级和顺序创建KD-Tree
        indices = np.arange(m)
        queue = deque([[0, 0, indices]])
        while queue:
            # 队列弹出的一项 树节点的索引，切分特征的索引，当前区域所有实例索引
            i, l, indices = queue.popleft()
            # 以实例第1个特征中位数作为切分点进行切分
            k = indices.size // 2
            indices = indices[np.argpartition(X[indices, l], k)]
            # 保存切分点实例到当前节点
            tree[i, 0] = indices[k]
            tree[i, 1] = l
            # 循环使用下一特征作为切分特征，
            l = (l + 1) % n
            # 将切分点左右区域的节点划分到左右子树：将实例索引入队，创建左右子树
            li, ri = 2*i+1, 2*i+2
            if indices.size > 1:
                queue.append([li, l, indices[:k]])
            if indices.size > 2:
                queue.append([ri, l, indices[k+1:]])
        return tree, tree_depth

    def _kd_tree_search(self, x, root, X, res_heap):
        '''搜索KD-Tree，将最近的k个邻居放入大端堆'''
        i = root
        idx = self.tree[i, 0]
        # 判断节点是否存在
        if idx < 0:
            return 
        
        # 获取当前root节点深度
        depth = self._node_depth(i)
        # 移动到x所在最小超矩形区域对应的叶节点
        for _ in range(self.tree_depth - depth):
            s = X[idx]
            l = self.tree[i, 1]
            if x[l] <= s[l]:
                i = i*2+1
            else:
                i = i*2+2
            idx = self.tree[i, 0]
        if idx > 0:
            # 计算到叶节点中实例的距离
            s = X[idx]
            d = np.linalg.norm(x - s)
            # 进行入堆出堆操作，更新当前k个最近邻居和最近距离
            heapq.heappushpop(res_heap, (-d, idx))
        while i > root:
            # 计算到父节点中实例的距离，并更新到当前最近距离
            parent_i = (i-1) // 2
            parent_idx = self.tree[parent_i, 0]
            parent_s = X[parent_idx]
            d = np.linalg.norm(x - parent_s)
            heapq.heappushpop(res_heap, (-d, parent_idx))
            l = self.tree[parent_i, 1]
            r = -res_heap[0][0]
            # 判断超球体(x,r)是否与兄弟节点的区域相交
            if np.abs(x[l] - parent_s[l]) < r:
                # 获取兄弟节点的树索引
                sibling_i = (i+1) if i%2 else (i-1)
                self._kd_tree_search(x, sibling_i, X, res_heap)
            # 递归向根节点回退
            i = parent_i

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.tree, self.tree_depth = self._kd_tree_build(X_train)

    def _predict_one(self, x):
        '''对单个实例进行预测'''
        # 创建存储k个最近邻居索引的最大堆
        # 注意：标准库中的heapq实现的是最小堆，以距离为负数作为键则等价于最大堆
        res_heap = [(-np.inf, -1)] * self.k_neighbors
        self._kd_tree_search(x, 0, self.X_train, res_heap)
        indices = [idx for _, idx in res_heap]
        counts = np.bincount(self.y_train[indices])
        return np.argmax(counts)

    def predict(self, X):
        return np.apply_along_axis(self._predict_one, axis = 1, arr = X)