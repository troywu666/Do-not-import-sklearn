'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-03 09:17:54
@LastEditors: Troy Wu
@LastEditTime: 2020-07-03 14:10:34
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
            res = self.kmeans(X)

        labels, self.centers_ = res
        return labels