'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-10 22:10:39
@LastEditors: Troy Wu
@LastEditTime: 2020-07-10 22:37:28
'''
import numpy as np

def svd(data):
    '''
    我们是如何知道仅需保留前3个奇异值的呢？
    确定要保留的奇异值的数目有很多启发式的策略，其中一个典型的做法就是保留矩阵中90%的能量信息。
    为了计算总能量信息，我们将所有的奇异值求其平方和。
    '''
    U, Sigma, Vt = np.linalg.svd(data)
    return Sigma