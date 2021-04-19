'''
@Description: 
@Version: 1.0
@Autor: Troy Wu
@Date: 2020-07-15 23:03:24
@LastEditors: Troy Wu
@LastEditTime: 2020-07-17 14:12:40
'''
import numpy as np

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue #名字变量
        self.count = numOccur #计数变量（频率）
        self.nodeLink = None #链接相似元素项
        self.parent = parentNode #当前父节点
        self.children = {} #用于存放子节点
        
    def inc(self, numOccur):
        '''
        给count变量增加定制
        '''
        self.count += numOccur

    def disp(self, ind = 1):
        '''
        将树以文本形式显示，方便调试
        '''
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
    ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
    ['z'],
    ['r', 'x', 'n', 'o', 's'],
    ['y', 'r', 'x', 'z', 'q', 't', 'p'],
    ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        fset = frozenset(trans)
        retDict.setdefault(fset, 0)
        retDict[fset] += 1
    return retDict

def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def updateTree(items, myTree, headerTable, count):
    if items[0] in myTree.children:
        myTree.children[items[0]].inc(count)
    else:
        myTree.children[items[0]] = treeNode(items[0], count, myTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = myTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], myTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1: ], myTree.children[items[0]], headerTable, count)

def createTree(dataSet, minSup = 1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + 1
    lessThanMinsup = list(filter(lambda k: headerTable[k] < minSup, headerTable.keys()))
    for k in lessThanMinsup:
        del (headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    myTree = treeNode('θ', 1, None)
    
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key = lambda x: (x[1], x[0]), reverse = True)]
            updateTree(orderedItems, myTree, headerTable, count)
    return myTree, headerTable

def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, headerTable):
    condPats = {}
    treeNode = headerTable[basePat][1]
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup = 1, preFix = set([]), freqItemList = []):
    bigL = [v[0] for v in sorted(headerTable.items(), keys = lambda x: (x[1][0], x[0]))]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable)
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            myCondTree.disp()
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)