# coding:utf-8
import pandas as pd
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


def calculateSampleDist(xi, xj):
    """
    计算两个样本间的距离
    @Parameters
        xi: 第i个样本
        xj: 第j个样本
    """
    s = math.sqrt(sum([pow((xi[i] - xj[i]), 2) for i in range(len(xi))]))
    return s


def calculateClusDist(data, cluster_i, cluster_j):
    """
    计算两个类间的距离
    @Parameters
        data:      原始数据
        cluster_i: 第i类的样本标号列表
        cluster_j: 第j类的样本标号列表
    """
    # 平均距离
    # s = sum([
    #     calculateSampleDist(list(data.loc[i]), list(data.loc[j]))
    #     for i in cluster_i for j in cluster_j
    # ])
    # d = s / (len(cluster_i) * len(cluster_j))
    # 最近距离
    s = [
        calculateSampleDist(list(data.loc[i]), list(data.loc[j]))
        for i in cluster_i for j in cluster_j
    ]
    d = min(s)
    return d


def agens(data, clusters):
    """
    由底向上的凝聚聚类
    @Parameters
        data:      原始数据
        clusters: 最大聚类数
    """
    clusList = [[i] for i in range(data.shape[0])]
    while len(clusList) != clusters:
        minDist = math.inf
        for cluster_i in clusList[:-1]:
            i = clusList.index(cluster_i)
            for cluster_j in clusList[i + 1:]:
                d = calculateClusDist(data, cluster_i, cluster_j)
                if d < minDist:
                    minDist = d
                    m, n = i, clusList.index(cluster_j)
        clusList[m].extend(clusList[n])
        del clusList[n]
        print(clusList)


if __name__ == "__main__":
    data = getData(fp="tra.csv")
    print(data)
    agens(data, 6)
