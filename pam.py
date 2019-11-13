# coding:utf-8
import pandas as pd
import math
import random

# 获取数据
def getData(fp):
    """
    @Parameters
        fp:   读取数据的文件名
    
    """
    df = pd.read_csv(fp)
    lables = df.keys().tolist()
    for lable in lables:
        df[lable] = preHand(df[lable])
    return df

# 数据标准化
def preHand(df):
    mean = df.mean()
    std = df.std()
    for i in range(df.shape[0]):
        df.loc[i] = (df.loc[i] - mean) / std
    return df

# 获取距离矩阵
def calculateDistMatrix(df):
    """
    计算该数据的距离矩阵/优异度矩阵
    @Parameters
        df:输入的数据
    """
    data = df.values.tolist()
    # print(data)
    matrix = []
    for xi in data:
        col = []
        for xj in data:
            s = math.sqrt(
                sum([pow((xi[i] - xj[i]), 2) for i in range(len(xi))]))
            col.append(s)
        matrix.append(col)
    return matrix

# 选取初始的簇中心点
def chooseInitPoint(data, clusters):
    """
    @Parameters
        data: 表示输入的数据，DataFrame型
        clusters:   表示用户所需要聚类的数目
    
    """
    initCenterPonit = []
    allPoint = [i for i in range(data.shape[0])]
    for i in range(clusters):
        j = random.choice(allPoint)
        initCenterPonit.append(j)
        del allPoint[allPoint.index(j)]
    return initCenterPonit

# 将每个点分配到最近的中心点所在的簇
def fenpei(distMatrix, initMedianPoint):
    exMedianPoint = [
        i for i in range(len(distMatrix)) if i not in initMedianPoint
    ]
    clusDict = dict(
        zip(initMedianPoint, [[] for i in range(len(initMedianPoint))]))
    # print(clusDict)
    # 分配非中心点到中心点的簇
    for i in exMedianPoint:
        mindist = math.inf
        index = 0
        for j in initMedianPoint:

            dist = distMatrix[i][j]
            if dist < mindist:
                mindist = dist
                index = j
        clusDict[index].append(i)
    return clusDict

# 计算损失值
def getCost(distMatrix, clusDict):
    oldCost = []
    for key, values in clusDict.items():
        oldCost.append(sum([distMatrix[key][v] for v in values]))
    return sum(oldCost)

# PAM算法
def pam(data, clusters, eps=0.2, maxIter=50):
    """
    中心点聚类
    @Parameters
        data:      原始数据
        clusters: 最大聚类数
        eps : 相邻两次更换中心点后的损失之差
        maxIter : 最大迭代次数
    """
    initMedianPoint = chooseInitPoint(data, clusters)
    print("初始中心点：", initMedianPoint)
    distMatrix = calculateDistMatrix(data)
    # 分配非中心点到中心点的簇
    clusDict = fenpei(distMatrix, initMedianPoint)
    # 计算原始代价
    oldCost = getCost(distMatrix, clusDict)
    lastCost = oldCost
    thisCost = 0
    # 更改中心点
    step = 1
    while True:
        minCost = math.inf
        for key, values in clusDict.items():
            changedMedianPoint = initMedianPoint.copy()
            index = initMedianPoint.index(key)
            for v in values:
                changedMedianPoint[index] = v
                aclusDict = fenpei(distMatrix, changedMedianPoint)
                newCost = getCost(distMatrix, aclusDict)
                if ((newCost - oldCost) < minCost):
                    newclusDict = aclusDict
                    thisCost = newCost
                    minCost = newCost - oldCost
                    # print(thisCost)
        clusDict = newclusDict
        initMedianPoint = [k for k in list(clusDict.keys())]
        print("新的中心点为：",initMedianPoint)
        step +=1
        if abs(lastCost - thisCost) <= eps or step == maxIter:
            return clusDict
        lastCost = thisCost

if __name__ == "__main__":
    df = getData(fp="tra.csv")
    clusDict = pam(df, 6, eps=0.5, maxIter=50)
    print(clusDict)
