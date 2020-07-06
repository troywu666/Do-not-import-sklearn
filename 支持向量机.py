'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-05 19:22:45
@LastEditors: Troy Wu
@LastEditTime: 2020-07-06 18:12:17
'''
import numpy as np

class SMO:
    def __init__(self, C, tol, kernel = 'rbf', gamma = None):
        # 惩罚系数
        self.C = C
        # 优化过程中的alpha步进阈值
        self.tol = tol
        if kernel == 'rbf':
            self.K = self._gaussian_kernel
            self.gamma = gamma
        else:
            self.kernel = self._linear_kernel
            
    def _linear_kernel(self, U, v):
        return np.dot(U, v)

    def _gaussian_kernel(self, U, v):
        if U.dim == 1:
            p = np.dot(U - v, U - v)
        else:
            p = np.sum((U - v) * (U - v), axis = 1)
        return np.exp(-p * self.gamma)

    def _g(self, x):
        alpha, b, X, y, E = self.args
        idx = np.nonzereos(alpha > 0)[0]
        if idx.size > 0:
            return np.sum(y[idx] * alpha[idx] * self.K(X[idx], x)) + b[0]
        return b[0]

    def _optimize_alpha_i(self, i, j):
        alpha, b, x, y, E = self.args
        C, tol, K = self.C, self.tol, self.K
        # 优化需有两个不同的alpha
        if i == j:
            return 0
        # 计算alpha[j]的边界
        if y[i] != y[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(C, C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[i] + alpha[j] - C)
            H = min(C, alpha[i] + alpha[j])
        
        # L=H时已无优化空间（一个点）
        if L == H:
            return 0

        # 计算eta
        eta = K[X[i], X[i]] + K[X[j], X[j]] - 2 * K[X[i], K[j]]
        if eta <= 0:
            return 0
        
        # 对于alpha非边界使用E缓存。边界alpha，动态计算E
        if 0 < alpha[i] < C:
            E_j = E[j]
        else: 
            E_j = self._g(X[j]) - y[j]
        # 计算alpha_j_new
        alpha_j_new = alpha[j] + y[j] * (E_i - E_j) / eta

        # 对alpha_j_new进行裁剪
        if alpha_j_new > H:
            alpha_j_new = H
        elif alpha_j_new < L:
            alpha_j_new = L
        alpha_j_new = np.round(alpha_j_new, 7)

        # 判断步长是否足够大
        if 

    def train(self, X_train, y_train):
        m, _ = X_train.shape
        alpha = np.zeros(m)
        b = np.zeros(1)
        # 创建E缓存
        E = np.zeros(m)
        self.args = [alpha, b, X_train, y_train, E]
        n_changed = 0
        examine_all = True
        while n_changed > 0 or examine_all:
            n_changed = 0
            # 迭代alpha_i
            for i in range(m):
                if examine_all or 0 < alpha[i] < self.C:
                    n_changed += self._optimize_alpha_i(i)
            # 若当前迭代非边界alpha，且没有alpha改变，下次迭代所有alpha。否则，下次迭代非边界alpha
