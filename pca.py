'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-10 21:17:32
@LastEditors: Troy Wu
@LastEditTime: 2020-07-10 21:36:57
'''
import numpy as np

def pca(dataMat, topNfeat = 9999):
    mean_value = np.mean(dataMat, axis = 0)
    mean_removed = dataMat - mean_value
    covMat = np.cov(mean_removed, rowvar = False)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[ : -(topNfeat + 1): -1]
    redEigVects = eigVects[: , eigValInd]
    lowDDataMat = mean_removed * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + mean_value
    return lowDDataMat, reconMat