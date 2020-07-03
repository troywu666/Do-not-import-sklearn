'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-03 09:17:54
@LastEditors: Troy Wu
@LastEditTime: 2020-07-03 16:40:16
'''
import numpy as np

class KMeans:
    def __init__(self, k_clusters, tol = 1e-4, max_iter = 300):
        self.k_clusters = k_clusters
        self.tol = tol
        self.max_iter = max_iter
        
    def _init_centers_random(self, X, k_clusters):
        _, n = X.shape
        xmin = np.min(X, axis = 0)
        xmax = np.max(X, axis = 0)
        return xmin + (xmax - xmin) * np.random.rand(k_clusters, n)

    def _kmeans(self, X):
        m, n = X.shape
        labels = np.zeros(m, dtype = np.int)
        distances = np.empty((m, self.k_clusters))
        centers_old = np.empty((self.k_clusters, n))
        centers = self._init_centers_random(X, self.k_clusters)
        for _ in range(self.max_iter):
            for i in range(self.k_clusters):
                np.sum((X - centers[i]) ** 2, axis = 1, out = distances[:, i])
            np.argmin(distances, axis = 1, out = labels)
            np.copyto(centers_old, centers)
            for i in range(self.k_clusters):
                cluster = X[labels == i]
                # 如果某个初始质心离所有数据都很远，可能导致没有实例被划入该簇，则返回None表示失败
                if cluster.size == 0:
                    return None
                np.mean(cluster,axis = 0, out = centers[i])
            delta_centers = np.sqrt(np.sum((centers - centers_old) ** 2, axis = 1))
            if np.all(delta_centers < self.tol):
                break
        return labels, centers

    def predict(self, X):
        res = None
        while not res:
            res = self._kmeans(X)

        labels, self.centers_ = res
        return labels

class KMeans_plus_plus:
    def __init__(self, k_clusters, tol, max_iter, n_init):
        self.k_clusters = k_clusters
        self.tol = tol
        self.max_iter = max_iter
        # 重新初始化质心点运行kmeans的次数
        self.n_init = n_init
        
    def _init_centers_kpp(self, X, n_clusters):
        m, n = X.shape
        distance = np.empty((m, n_clusters - 1))
        centers = np.empty((n_clusters, n))
        np.copyto(centers[0], X[np.random.randint(m)])
        for j in range(1, n_clusters):
            for i in range(j):
                np.sum((X - centers[0]) ** 2, axis = 1, out = distance[:, i])
            # 计算各点到最近中心的距离的平方
            nds = np.min(distance[:, : j], axis = 1)
            # 以各点最近中心的距离的平方构成的加权概率分布，产生下一个簇中心
            r = np.sum(nds) * np.random.random()
            for k in range(m):
                r -= nds[k]
                if r < 0:
                    break
            np.copyto(centers[j], X[k])
        return centers
    
    def _kmeans(self, X):
        m, n = X.shape
        labels = np.zeros(m, dtype = np.int) 
        distances = np.empty((m, self.k_clusters))
        centers_old = np.empty((self.k_clusters, n)) 
        centers = self._init_centers_kpp(X, self.k_clusters)
        for _ in range(self.max_iter):
            for i in range(self.k_clusters):
                np.sum((X - centers[i]) ** 2, axis = 1, out = distances[:, i])
            np.argmin(distances, axis = 1, out = labels)     
            np.copyto(centers_old, centers)
            for i in range(self.k_clusters):
                cluster = X[labels == i]
                if cluster.size == 0:
                    return None
                np.mean(cluster, axis = 0, out = centers[i])
            delta_centers = np.sqrt(np.sum((centers - centers_old) ** 2, axis = 1))
            if delta_centers < self.tol:
                break
        
    def predict(self, X):
        result = np.empty((self.n_init, 3), dtype = np.object)
        for i in range(self.n_init):
            res = None
            while res is None:
                res = self._kmeans(X)
            result[i] = res
        k = np.argmin(result[:, -1])
        labels, self.centers_, self.sse_ = result[k]
        return labels