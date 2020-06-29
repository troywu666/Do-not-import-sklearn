'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-06-29 11:15:05
@LastEditors: Troy Wu
@LastEditTime: 2020-06-29 20:43:48
'''
import numpy as np

class OLSLinearRegression:
    '''基于最小二乘法的线性回归'''
    def _ols(self, X, y):
        '''最小二乘法估算W'''
        tmp = np.linalg.inv(np.matmul(X.T, X))
        tmp = np.matmul(tmp, X.T)
        return np.matmul(tmp, y)
    
    def _preprocess_data_X(self, X):
        '''数据预处理'''
        m, n = X.shape
        X_ = np.empty((m, n+1))
        X_[:, 0] = 1
        X_[:, 1: ] = X
        return X_

    def train(self, X_train, y_train):
        '''训练模型'''
        X_train = self._preprocess_data_X(X_train)
        self.w = self._ols(X_train, y_train)
        
    def predict(self, X):
        '''预测'''
        X = self._preprocess_data_X(X)
        return np.matmul(X, self.w)

class GDLinearRegression:
    '''批量梯度下降'''
    def __init__(self, n_iter = 200, eta = 1e-3, tol = None):
        # 训练迭代次数
        self.n_iter = n_iter
        # 学习率
        self.eta = eta
        # 误差变化阈值
        self.tol = tol
        # 模型参数w（训练时初始化）
        self.w = None

    def _loss(self, y, y_pred):
        '''计算损失'''
        return np.sum((y_pred - y) ** 2) / y.size

    def _gradient(self, X, y ,y_pred):
        '''计算梯度'''
        return np.matmul(y_pred - y, X) / y.size

    def _gradient_descent(self, w, X, y):
        '''梯度下降算法'''
        if self.tol is not None:
            loss_old = np.inf
        # 使用梯度下降，至多迭代n_iter次，更新w
        for step_i in range(self.n_iter):
            y_pred = self._predict(X, w)
            # 计算损失
            loss = self._loss(y, y_pred)
            print('Epoch is {}. The loss is {}'.format(step_i, loss))
            
            # 早期停止法
            if self.tol is not None:
                if loss_old - loss < self.tol:
                    break
                loss_old = loss
            # 计算梯度
            grad = self._gradient(X, y, y_pred)
            # 更新参数
            w -= self.eta * grad

    def _preprocess_data_X(self, X):
        '''数据预处理'''
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1: ] = X
        return X_

    def train(self, X_train, y_train):
        '''训练'''
        X_train = self._preprocess_data_X(X_train)
        # 初始化参数向量w
        _, n = X_train.shape
        self.w = np.random.random(n) * 0.5
        
        # 执行梯度下降训练w
        self._gradient_descent(self.w, X_train, y_train)
    
    def _predict(self, X, w):
        '''预测内部接口'''
        return np.matmul(X, w)
    
    def predict(self, X):
        X = self._preprocess_data_X(X)
        return self._predict(X, self.w)

class SGDLinearRegression:
    '''随机梯度下降'''
    def __init__(self, n_iter = 1000, eta = 0.01):
        # 训练迭代次数
        self.n_iter = n_iter
        # 学习率
        self.eta = eta
        # 模型参数theta(训练时根据数据特征量初始化)
        self.theta = None

    def _gradient(self, xi, yi, theta):
        # 计算当前梯度
        return -xi * (yi - np.dot(xi, theta))
    
    def _stochastic_gradient_descent(self, X, y, eta, n_iter):
        # 复制X（避免随机乱序时改变原X
        X = X.copy()
        m, _ = X.shape
        eta0 = eta
        step_i = 0

        for _ in range(n_iter):
            # 随机乱序
            np.random.shuffle(X)
            for i in range(m):
                grad = self._gradient(X[i], y[i], self.theta)
                # 更新参数theta
                step_i += 1
                eta = eta0 / np.power(step_i, 0.25)
                self.theta += eta * -grad

    def train(self, X, y):
        if self.theta is None:
            _, n = X.shape
            self.theata = np.zeros(n)
        self._stochastic_gradient_descent(X, y, self.eta, self.n_iter)
    
    def predict(self, X):
        return np.dot(X, self.theta)

class LogisticRegression:
    def __init__(self, n_iter = 200, eta = 1e-3, tol = None):
        # 训练迭代次数
        self.n_iter = n_iter
        # 学习率
        self.eta = eta
        # 误差变换阈值
        self.tol = tol
        # 模型参数（训练时初始化）
        self.w = None

    def _z(self, X, w):
        '''计算x与w的內积'''
        return np.dot(X, w)

    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-z)) 

    def _predict_proba(self, X, w):
        z = self._z(self, X, w)
        return self._sigmoid(z)

    def _loss(self, y, y_proba):
        m = y.size
        p = y_proba**y + (1-y_proba)**(1-y)
        return -np.sum(np.log(p)) / m

    def _gradient(self, X, y, y_proba):
        return np.matmul(y_proba - y, X) / y.size

    def _gradient_descent(self, w, X, y):
        if self.tol is not None:
            loss_old = np.inf
        for step_i in range(self.n_iter):
            y_proba = self._predict_proba(X, w)
            loss = self._loss(y, y_proba)
            print('Epoch is {}. The loss is {}'.format(step_i, loss))

            if self.tol is not None:
                if loss_old - loss < self.tol:
                    break
                loss_old = loss

            grad = self._gradient(X, y, y_proba)
            w -= self.eta * grad
    
    def _preprocess_data_X(self, X):
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def train(self, X_train, y_train):
        X_train = self._preprocess_data_X(X_train)
        _, n = X_train.shape
        self.w = np.random.random(n) * 0.05
        self._gradient_descent(self.w, X_train, y_train)
        
    def predict(self, X):
        X = self._preprocess_data_X(X)
        y_pred = self._predict_proba(X, self.w)
        return np.where(y_pred > 0.5, 1, 0)

class SoftmaxRegression:
    def __init__(self, n_iter = 200, eta = 1e-3, tol = None):
        self.n_iter = n_iter
        self.eta = eta
        self.tol = tol
        self.W = None

    def _z(self, X, W):
        if X.ndim == 1:
            return np.dot(W,X)
        return np.matmul(X, W)

    def _softmax(self, Z):
        E = np.exp(Z)
        if Z.ndim == 1:
            return E / sum(E)
        return E / np.sum(E, axis = 1, keepdims = True)
        
    def _predict_proba(self, X, W):
        Z = self._z(X, W)
        return self._softmax(Z)

    def _loss(self, y, y_proba):
        m = y.size
        p = y_proba[range(m), y]
        return -np.sum(np.log(p)) / m

    def _gradient(self, xi, yi, yi_proba):
        K = yi_proba.size
        y_bin = np.zeros(K)
        y_bin[yi] = 1
        return (yi_proba - y_bin)[:, None] * xi

    def _stochastic_gradient_descent(self, W, X, y):
        if self.tol is not None:
            loss_old = np.inf
            end_count = 0

        m = y.size
        idx = np.arange(m)
        for step_i in range(self.n_iter):
            y_proba = self._predict_proba(X, W)
            loss = self._loss(y, y_proba)
            print('Epoch is {}. The loss is {}'.format(step_i, loss))
            
            if self.tol is not None:
                if loss_old - loss < self.tol:
                    # 随机梯度下降的loss曲线不像批量梯度下降那么平滑（上下起伏），因此需要连续多次下降不足阈值，才终止迭代
                    end_count += 1
                    if end_count == 5:
                        break
                else:
                    end_count = 0
                loss_old = loss
                
            np.random.shuffle(idx)
            for i in idx:
                yi_proba = self._predict_proba(X[i], W)
                grad = self._gradient(X[i], y[i], yi_proba)
                W -= self.eta * grad

    def _preprocess_data_X(self, X):
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def train(self, X_train, y_train):
        X_train = self._preprocess_data_X(X_train)
        k = np.unique(y_train).size
        _, n = X_train.shape
        self.W = np.random.random((k, n)) * 0.05
        self._stochastic_gradient_descent(self.W, X_train, y_train)

    def predict(self, X):
        X = self._preprocess_data_X(X)
        Z = self._z(X, self.W)
        return np.argmax(Z, axis = 1)