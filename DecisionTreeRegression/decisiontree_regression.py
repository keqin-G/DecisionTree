import csv
import matplotlib.pyplot as plt
from math import inf
import plotTree
import random


def createTree(dataset, labels, max_depth = inf, min_sample_split = 2):
    """
        :param dataset: 数据集
        :param labels: 特征集
        :param max_depth: 最大深度
        :param min_sample_split: 最小拆分数
        
        :return: 回归决策树
    """
    # 获取所有样本的值
    target_values = [e[-1] for e in dataset]
    
    # 如果所有样本的值都相同 或 达到最大深度 或 样本数量小于最少拆分数，返回平均值
    if len(set(target_values)) == 1 or max_depth <= 0 or len(dataset) <= min_sample_split:
        return sum(target_values) / len(target_values)
    
    # 如果所有特征都已经用于划分 返回平均值
    if len(dataset[0]) == 1:
        return sum(target_values) / len(target_values)
    
    # 选择最好的特征和阈值
    bestFeature, bestThreshold = chooseBestFeatureToSplit(dataset)
    # 如果找不到一个最好特征 返回平均值
    if bestFeature == -1:
        return sum(target_values) / len(target_values)
    # 选中的特征的标签
    bestFeatureLabel = labels[bestFeature]
    # 将选中的特征从特征集中删除 
    del labels[bestFeature] 
    # 以选中的特征为根节点创建树
    tree = {bestFeatureLabel: {'threshold': bestThreshold}} 
    # 划分数据集
    subDataset1, subDataset2 = splitContinuousDataset(dataset, bestFeature, bestThreshold) 
    # 递归创建左右子树
    tree[bestFeatureLabel]['<='] = createTree(subDataset1, labels[:], max_depth - 1, min_sample_split) 
    tree[bestFeatureLabel]['>'] = createTree(subDataset2, labels[:], max_depth - 1, min_sample_split)
    return tree

def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1 
    baseMSE = calcMSE(dataset)
    bestInfoGain = 0
    bestFeature = -1
    bestThreshold = 0
    for i in range(numFeatures):
        featureList = [e[i] for e in dataset]
        uniqueVals = set(featureList)
        uniqueVals = sorted(uniqueVals)
        # 取相邻两个值的中间值作为阈值
        threshold = [(uniqueVals[i] + uniqueVals[i + 1]) / 2 for i in range(len(uniqueVals) - 1)]
        for val in threshold:
            subDataset1, subDataset2 = splitContinuousDataset(dataset, i, val)
            p = len(subDataset1) / float(len(subDataset1) + len(subDataset1))
            # 实测两种方法差不多 但加权的方式极差小一点
            mse = p * calcMSE(subDataset1) + (1 - p) * calcMSE(subDataset2)
            # mse = calcMSE(subDataset1) + calcMSE(subDataset2) 
            infoGain = baseMSE - mse
            # 更新最佳信息增益
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
                bestThreshold = val
    return bestFeature, bestThreshold

def splitContinuousDataset(dataset, index, value):
    # 以index为特征 划分数据集
    subDataset1 = [sample[:index] + sample[index + 1:] for sample in dataset if sample[index] <= value]
    subDataset2 = [sample[:index] + sample[index + 1:] for sample in dataset if sample[index] > value]
    return subDataset1, subDataset2

def calcMSE(dataset):
    # 计算均方误差
    num = len(dataset)
    target_values = [e[-1] for e in dataset]
    avg = sum(target_values) / float(num)
    mse = sum([(target - avg) ** 2 for target in target_values])
    return mse

def createDataset(filename='data.csv'):
    # 读取数据集
    dataset = []
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            row_data = []
            # 跳过所有带na的数据
            if row.count('NA') > 0:
                continue
            for val in row:
                try:
                    val = float(val)
                    row_data.append(val)
                except TypeError and ValueError:
                    row_data.append(val)
            dataset.append(row_data)
    labels = dataset[0][:-1]
    dataset = dataset[1:]  
    return dataset, labels

def predict(tree, labels, test_data):
    res = []
    for data_point in test_data:
        prediction = traverseTree(tree, labels, data_point)
        res.append(prediction)
    return res

def traverseTree(tree, labels, data_point):
    if not isinstance(tree, dict):
        return tree

    feature_label = list(tree.keys())[0]
    threshold = tree[feature_label]['threshold']

    feature_index = labels.index(feature_label)
    feature_value = data_point[feature_index]

    if feature_value <= threshold:
        branch_key = '<='
    else:
        branch_key = '>'
    return traverseTree(tree[feature_label][branch_key], labels, data_point)

if __name__ == '__main__':
    dataset, labels = createDataset('HousingData.csv')
    feature_labels = labels[:]
    dataset_size = len(dataset)
    test_data_size = 100
    t = 0
    test_data = []
    training_data = []
    for data in dataset:
        i = random.randint(0, dataset_size - 1)
        if i & 1 and t < test_data_size:
            t += 1
            test_data.append(data)
        else:
            training_data.append(data)

    tree = createTree(training_data, labels, max_depth = inf)    
    # plotTree.createPlot(tree)

    real = [e[-1] for e in test_data]
    predictions = predict(tree, feature_labels, test_data)
    r = [(real[i] - predictions[i]) ** 2 for i in range(test_data_size)]
    r = sum(r) / test_data_size
    print(f'mse: {r}')
    plt.scatter([i for i in range(test_data_size)], real, color = 'blue', s = 10)
    plt.scatter([i for i in range(test_data_size)], predictions, color = 'red', s = 10)
    plt.show()