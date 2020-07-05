'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-05 11:04:07
@LastEditors: Troy Wu
@LastEditTime: 2020-07-05 19:02:39
'''
import numpy as np

class ANNClassifier:
    def __init__(self, hidden_layer_sizes = (30, 30), eta = 0.01, max_iter = 500, tol = 0.001):
        '''构造器'''
        # 各隐藏层节点的个数
        self.hidden_layer_sizes = hidden_layer_sizes
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol

    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-z))
 
    def _z(self, x, W):
        return np.matmul(x, W)

    def _error(self, y, y_predict):
        return np.sum((y - y_predict)**2) / len(y)

    def _backpropagation(self, X, y):
        '''反向传播算法（基于随机梯度下降）'''
        m, n = X.shape
        _, n_out = y.shape
        # 获得各层节点个数元组layer_sizes以及总层数layer_n
        layer_sizes = self.hidden_layer_sizes + (n_out,)
        layer_n = len(layer_sizes)

        # 对于每一层，将所有节点的权向量（以列向量形式）存为一个矩阵，保存至W_list
        W_list = []
        li_size = n
        for lj_size in layer_sizes:
            W = np.random.rand(li_size + 1, lj_size) * 0.05
            W_list.append(W)
            li_size = lj_size
        
        # 创建运行梯度下降时所用的列表
        in_list = [None] * layer_n
        z_list = [None] * layer_n
        out_list = [None] * layer_n
        delta_list = [None] * layer_n

        # 随机梯度下降
        idx = np.arange(m)
        for _ in range(self.max_iter):
            X, y = X[idx], y[idx]
            for x, t in zip(X, y):
                # 第i-1层输出添加x0 = 1，作为第i层输入
                out = x
                for i in range(layer_n):
                    in_ = np.ones(out.size + 1)
                    in_[1: ] = out
                    # 计算第i层所有节点的净输入
                    z = self._z(in_, W_list[i])
                    # 计算第i层各节点输出值
                    out = self._sigmoid(z)
                    # 保存第i层各节点的输入，净输入，输出
                    in_list[i], z_list[i], out_list[i] = in_, z, out      
                # 反向传播计算各层的delta
                # 输出层
                delta_list[-1] = out*(1. - out) * (t - out)  
                # 隐藏层
                for i in range(layer_n - 2, -1, -1):
                    out_i, W_j, delta_j = out_list[i], W_list[i+1], delta_list[i+1]
                    delta_list[i] = out_i * (1. - out_i) * np.matmul(W_j[1:], delta_j[:, None]).T[0]
                    # 更新所有节点的权重
                    for i in range(layer_n):
                        in_i, delta_i = in_list[i], delta_list[i]
                        W_list[i] += in_i[:, None] * delta_i * self.eta
                # 计算训练误差
                y_pred = self._predict(X, W_list)
                err = self._error(y, y_pred)

                # 判断收敛（误差是否小于阈值）
                if err < self.tol:
                    break
                print('{},err:{}'.format(_+1, err))
        return W_list

    def train(self, X, y):
        self.W_list = self._backpropagation(X, y)
    
    def _predict(self, X, W_list, return_bin = False):
        layer_n = len(W_list)
        out = X
        for i in range(layer_n):
            m, n = out.shape
            in_ = np.ones((m, n+1))
            in_[:, 1:] = out
            z = self._z(in_, W_list[i])
            out = self._sigmoid(z)

        if return_bin:
            idx = np.argmax(out, axis = 1)
            out_bin = np.zeros_like(out)
            out_bin[range(len(idx)), idx] = 1
            return out_bin
        return out

    def predict(self, X):
        return self._predict(X, self.W_list, return_bin = True)

class ANNRegressor:
    def __init__(self, hidden_layer_sizes = (30, 30), eta = 0.01, max_iter = 500, tol = 0.001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
    
    def _sigmoid(self, z):
        return 1. /(1. + np.exp(-z))

    def _z(self, x, W):
        return np.matmul(x, W)
    
    def _error(self, y, y_predict):
        return np.sum((y - y_predict) ** 2) / len(y)

    def _backpropagation(self, X, y):
        m, n = X.shape
        _, n_out = y.shape
        layer_sizes = self.hidden_layer_sizes + (n_out,)
        layer_n = len(layer_sizes)
        W_list = []
        li_size = n
        for lj_size in layer_sizes:
            W = np.random.rand((li_size + 1, lj_size)) * 0.05
            W_list.append(W)
            li_size = lj_size
            
        in_list = [None] * layer_n
        z_list = [None] * layer_n
        out_list = [None] * layer_n
        delta_list = [None] * layer_n
        
        idx = np.arange(m)
        for _ in range(self.max_iter):
            np.random.shuffle(idx)
            X, y = X[idx], y[idx]
            for x, t in zip(X, y):
                out = x
                for i in range(layer_n):
                    in_ = np.ones(out.size + 1)
                    in_[1:] = out
                    z = self._z(in_, W_list[i])
                    if i != layer_n - 1:
                        out = self._sigmoid(z)
                    else:
                        out = z
                    in_list[i], z_list[i], out_list[i] = in_, z, out
                    
                delta_list[-1] = t - out
                for i in range(layer_n - 2, -1, -1):
                    out_i, W_j, delta_j = out_list[i], W_list[i + 1], delta_list[i + 1]
                    delta_list[i] = out_i * (1. - out_i) * np.matmul(W_j[1:], delta_j[:, None]).T[0]

            for i in range(layer_n):
                in_i, delta_i = in_list[i], delta_list[i]
                W_list[i] += in_i[:, None] * delta_i * self.eta
            y_pred = self._predict(X, W_list)
            err = self._error(y, y_pred)

            if err < self.tol:
                break
        return W_list

    def train(self, X, y):
        self.W_list = self._backpropagation(X, y)
    
    def _predict(self, X, W_list):
        layer_n = len(W_list)
        out = X
        for i in range(layer_n):
            m, n = out.shape
            in_ = np.ones((m, n + 1))
            in_[: , 1:] = out
            z = self._z(in_, self.W_list)
            if i != layer - 1:
                out = self._sigmoid(z)
            else:
                out = z
        return out

    def predict(self, X):
        return self._sigmoid(X, self.W_list)