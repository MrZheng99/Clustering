# coding:utf-8
import pandas as pd
import random
import math


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


def preHand(df):
    mean = df.mean()
    std = df.std()
    for i in range(df.shape[0]):
        df.loc[i] = (df.loc[i] - mean) / std
    return df


def chooseInitCenter(data, k):
    """
    @Parameters
        data: 表示输入的数据，DataFrame型
        K:   表示用户所需要聚类的数目
    
    """
    initCenterPonit = []
    allPoint = [i for i in range(data.shape[0])]
    for i in range(k):
        j = random.choice(allPoint)
        initCenterPonit.append(data.loc[j].tolist())
        del allPoint[allPoint.index(j)]
    return initCenterPonit


def calculateDist(xi, xj):
    """
    计算两个样本间的距离,xj一般表示中心点，计算第Xi个样本到中心点xj的距离
    @Parameters
        xi: 第i个样本
        xj: 第j个样本
    """
    s = 0
    for i in range(len(xi)):
        s += pow((xi[i] - xj[i]), 2)
    return math.sqrt(s)


def signClass(data, CenterPonit):
    """
    将样本划分给距离此样本最近的类别
    @Parameters
    data:数据集
    CenterPoint:中心点
    """
    dataClassDict = {}
    cols = data.shape[0]
    for i in range(cols):
        signLabel = frozenset(CenterPonit[0])
        minDist = math.inf
        for a in CenterPonit:
            dist = calculateDist(data.loc[i], a)
            if dist < minDist and dist != 0:
                signLabel = frozenset(a)
                minDist = dist
        if dataClassDict.get(signLabel, None) != None:
            dataClassDict[signLabel].append(i)
        else:
            dataClassDict[signLabel] = [i]
    return dataClassDict


def updateCenterPoint(data, dataClassDict):
    """
    更新每个类别的中心点，中心点为隶属于该类别的所有样本均值
    @Parameters
    dataClassDict:上一次的聚类结果
    """
    centerPoint = []
    rows = data.shape[1]
    for key, values in dataClassDict.items():
        n = 1 + len(values)
        s = list(key).copy()
        for v in list(values):
            for i in range(rows):
                s[i] += data.loc[v][i]
        s = [e / n for e in s]
        centerPoint.append(s)
    return centerPoint


def MSE(data, dataClassDict, CenterPonit):
    """
    判断是否终止聚类，终止条件：最大迭代次数，最小误差平方MSE
    @Parameters
        dataClassDict：上一次聚类的结果
        CenterPonit ：新的中心点
    """
    mse = 0
    j = 0
    for k in list(dataClassDict.keys()):
        values = list(dataClassDict[k])
        s = 0
        for v in values:
            s = s + calculateDist(data.loc[v], CenterPonit[j])
        mse += s
        j += 1
    return mse


def kmeans(data, max_iter, min_mse, clusters=3):
    """
    kmeans算法聚类
    @Parameters
        data ：原始数据
        max_iter ：最大迭代次数
        min_mse ：最小的误差平方和
        clusters:聚类数
    """
    initCenterPonit = chooseInitCenter(data=data, k=clusters)
    step = 0
    while True:
        dataClassDict = signClass(data, initCenterPonit)
        initCenterPonit = updateCenterPoint(data, dataClassDict)
        if MSE(data, dataClassDict,
               initCenterPonit) < min_mse or step == max_iter:
            return dataClassDict
        step += 1


if __name__ == "__main__":
    data = getData(fp="tra.csv")
    dataClassDict = kmeans(data, max_iter=100, min_mse=1, clusters=7)
    print(dataClassDict)
