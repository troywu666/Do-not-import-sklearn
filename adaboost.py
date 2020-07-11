'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-11 21:09:27
@LastEditors: Troy Wu
@LastEditTime: 2020-07-12 00:33:50
'''
import numpy as np
import pandas as pd

def get_Mat(path):
    dataSet = pd.read_table(path, header = None)
    xMat = np.mat(dataSet.iloc[:, : -1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    return xMat, yMat

def Classify0(xMat, i, Q, S):
    re = np.ones((xMat.shape[0], 1))
    if S == 'lt':
        re[xMat[:, i] <= Q] = -1 # 如果小于阈值，则赋值为-1
    else:
        re[xMat[:, i] > Q] = -1 # 如果大于阈值，则赋值为-1
    return re

def get_Stump(xMat, yMat, D):
    m, n = xMat.shape
    Steps = 10
    bestStump = {}
    bestClas = np.mat(np.zeros((m, 1)))
    minE = np.inf # 最小误差
    for i in range(n):
        Min = xMat[:, i].min()
        Max = xMat[:, i].max()
        stepSize = (Max - Min) / Steps
        for j in range(-1, int(Steps)+1):
            for S in ['lt', 'gt']:
                Q = (Min + j*stepSize)
                re = Classify0(xMat, i, Q, S)
                err = np.mat(np.ones((m, 1)))
                err[re == yMat] = 0 # 分类正确的赋值为0
                eca = D.T * err
                if eca < minE:
                    minE = eca
                    bestClas = re.copy()
                    bestStump['特征列'] = i
                    bestStump['阈值'] = Q
                    bestStump['标志'] = S
    return bestStump, minE, bestClas

def Ada_train(xMat, yMat, maxC = 40):
    '''
    输入：
        maxC：最大迭代次数
    返回：
        weakClass：弱分类器信息
        aggClass：类别估计值（其实就是更改了标签的估计值）
    '''
    weakClass = []
    m = xMat.shape[0]
    D = np.mat(np.ones((m, 1)) / m) # 初始化权重
    aggClass = np.mat(np.zeros((m, 1)))
    for i in range(maxC):
        Stump, error, bestClas = get_Stump(xMat, yMat, D) # 构建单层决策树
        alpha = float(0.5*np.log((1-error) / max(error, 1e-16)))
        Stump['alpha'] = np.round(alpha, 2)
        weakClass.append(Stump)
        expon = np.multiply(-1 * alpha * yMat, bestClas)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClass += alpha*bestClas
        aggErr = np.multiply(np.sign(aggClass) != yMat, np.ones((m, 1)))
        errRate = aggErr.sum()
        if errRate == 0:
            break
    return weakClass, aggClass

def AdaClassify(data, weakClass):
    dataMat = np.mat(data)
    m = dataMat.shape[0]
    aggClass = np.mat(np.zeros((m, 1)))
    for i in range(len(weakClass)):
        classEst = Classify0(dataMat, \
                            weakClass[i]['特征列'],
                            weakClass[i]['阈值'],
                            weakClass[i]['标志'])
        aggClass += weakClass[i]['alpha'] * classEst
    return np.sign(aggClass)