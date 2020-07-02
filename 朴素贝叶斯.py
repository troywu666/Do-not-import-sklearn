'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-01 16:23:33
@LastEditors: Troy Wu
@LastEditTime: 2020-07-02 15:41:01
'''
import numpy as np

class BernoulliNavieBayes:
    def __init__(self, alpha = 1.):
        # 平滑系数，默认为1（拉普拉斯平滑）
        self.alpha = alpha

    def _class_prior_proba_log(self, y, classes):
        '''计算所有类别的先验证概率'''
        # 统计各样本的样本数量
        c_count = np.count_nonzero(y == classes[:, None], axis = 1)
        # 计算各类别的先验概率（平滑修正）
        p = (c_count + self.alpha) / (len(y) + self.alpha * len(classes))
        return np.log(p)

    def _conditional_proba_log(self, X, y, classes):
        '''计算所有的条件概率'''
        _, n = X.shape
        K = len(classes)
        # P_log：2个条件概率的对数的
        # 矩阵P_log[0]存储所有log(P(x^(j)=0|y=c_k))
        # 矩阵P_log[1]存储所有log(P(x^(j)=1|y=c_k))
        P_log = np.empty((2, K, n))
        # 迭代每一个类别c_k
        for k, c in enumerate(classes):
            X_c = X[y == c]
            # 统计特征值为1的实例的数量
            count1 = np.count_nonzero(X_c, axis = 0)
            # 计算条件概率
            p1 = (count1 + self.alpha) / (len(X_c) + 2 * self.alpha)
            # 将log(P(x^(j)=0|y=c_k))和log(P(x^(j)=1|y=c_k))存入矩阵
            P_log[0, k] = np.log(1- p1)
            P_log[1, k] = np.log(p1)
        return P_log

    def train(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.pp_log = self._class_prior_proba_log(y_train, self.classes)
        self.cp_log = self._conditional_proba_log(X_train, y_train, self.classes)
        
    def _predict_one(self, x):
        K = len(self.classes)
        p_log = np.empty(K)
        idx1 = x==1
        idx0 = ~idx1
        for k in range(K):
            p_log[k] = self.pp_log[k] + np.sum(self.cp_log[0, k][idx0]) + np.sum(self.cp_log[0, k][idx1])

        return np.argmax(p_log)

    def predict(self, X):
        return np.apply_along_axis(self._predict_one, axis = 1, arr = X)