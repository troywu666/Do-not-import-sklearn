'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-15 15:20:22
@LastEditors: Troy Wu
@LastEditTime: 2020-07-17 15:49:41
'''
import numpy as np

class Apriori:
    def __init__(self, minSupport, minConfidence, data):
        # 最小支持度
        self.minSupport = minSupport
        # 最小置信度
        self.minConfidence = minConfidence
        self.data = data

    # 用于从候选项集Ck生成Lk，Lk表示满足最低支持度的元素集合
    def createC1(self, data):
        C1 = list() # C1为大小为1的项的集合
        for items in data:
            for item in items:
                if [item] not in C1:
                    C1.append([item])
        return list(map(frozenset, sorted(C1)))
    
    def scanD(self, Ck):
        Data = list(map(set, self.data))
        CkCount = {}
        for items in Data:
            for one in Ck:
                if one.issubset(items):
                    CkCount.setdefault(one, 0)
                    CkCount[one] += 1
        numItems = len(list(Data))
        Lk = []
        supportData = {}
        for key in CkCount:
            support = CkCount[key] * 1.0 / numItems
            if support >= self.minSupport:
                Lk.insert(0, key)
            supportData[key] = support
        return Lk, supportData
    
    # k是项集元素个数k
    def generateNewCk(self, Lk, k):
        nextLk = []
        lenLk = len(Lk)
        # 若两个项集的长度为k-1,则必须前k-2项相同才可连接，即求并集
        for i in range(lenLk):
            for j in range(i+1, lenLk):
                L1 = list(Lk[i])[: k-2]
                L2 = list(Lk[j])[: k-2]
                if sorted(L1) == sorted(L2):
                    nextLk.append(Lk[i] | Lk[j])
        return nextLk
    
    # 生成频繁项集
    def generateLk(self):
        C1 = self.createC1(self.data)
        L1, supportData = self.scanD(C1)
        L = [L1]
        k = 2
        while len(L[k - 2]) > 0:
            Ck = self.generateNewCk(L[k - 2], k)
            Lk, supK = self.scanD(Ck)
            supportData.update(supK)
            L.append(Lk)
            k += 1
        return L, supportData

    def generateRules(self, L, supportData):
        ruleResult = [] # 最终记录的关联规则结果
        for i in range(1, len(L)):
            for ck in L[i]:
                Cks = [frozenset([item]) for item in ck]
                self.rulesOfMore(ck, Cks, supportData, ruleResult)
        return ruleResult

    def rulesOfTwo(self, ck, Cks, supportData, ruleResult):
        prunedH = []
        for oneCk in Cks:
            conf = supportData[ck] / supportData[ck - oneCk]
            if conf >= self.minConfidence:
                print(ck - oneCk, '-->', oneCk, 'Confidence is: ', conf)
                ruleResult.append((ck - oneCk, oneCk, conf))
                prunedH.append(oneCk)
        return prunedH

    def rulesOfMore(self, ck, Cks, supportData, ruleResult):
        m = len(Cks[0])
        while len(ck) > m:
            Cks = self.rulesOfTwo(ck, Cks, supportData, ruleResult)
            if len(Cks) > 1:
                Cks = self.generateNewCk(Cks, m+1)
                m += 1
            else:
                break