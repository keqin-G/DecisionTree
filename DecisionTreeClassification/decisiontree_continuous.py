from math import log
import numpy
import plotTree
import csv
import matplotlib.pyplot as plt


'''
这里主要修改了 chooseBestFeatureToSplit 函数，
以及在 createTree 函数中添加了对连续特征的处理。
在 chooseBestFeatureToSplit 函数中，对于连续特征，
使用了阈值来划分数据集。在 createTree 函数中，
使用了 <= 和 > 来表示特征值与阈值的关系。
这样，在树的预测阶段，可以根据特征值与阈值的比较结果来遍历树的分支，
直到达到叶子节点。
'''

def createTree(dataset, labels, max_depth=float('inf'), min_samples_split=2):
    """
    递归构建决策树

    :param dataset (list): 数据集
    :param labels (list): 特征标签列表
    :param feature_labels (list): 存储构建过程中使用的特征标签
    :param max_depth (float): 树的最大深度
    :param min_samples_split (int): 最小样本拆分数

    :return:
       dict: 决策树
    """
    classList = [e[-1] for e in dataset]
    if classList.count(classList[0]) == len(classList) or max_depth == 0 or len(dataset) < min_samples_split:
        # 如果所有样本属于同一类别，达到最大深度，或样本数量小于最小拆分数，则返回该类别
        return majorityCnt(classList)
    if len(dataset[0]) == 1:
        return majorityCnt(classList)  # 如果所有特征都用于划分，返回样本中类别最多的类别
    # if not labels:
    #     return majorityCnt(classList)
    bestFeature, bestThreshold = chooseBestFeatureToSplit(dataset)
    bestFeatureLabel = labels[bestFeature]
    tree = {bestFeatureLabel: {"threshold": bestThreshold}}
    del labels[bestFeature]
    subDataset1, subDataset2 = splitContinuousDataset(dataset, bestFeature, bestThreshold)
    subLabels1, subLabels2 = labels[:], labels[:]
    tree[bestFeatureLabel]["<="] = createTree(subDataset1, subLabels1, max_depth - 1, min_samples_split)
    tree[bestFeatureLabel][">"] = createTree(subDataset2, subLabels2, max_depth - 1, min_samples_split)
    return tree

def majorityCnt(class_list):
    """
    计算类别的多数投票z

    :param class_list (list): 类别列表

    :return
            str: 多数类别
    """
    classCounter = {}
    for i in class_list:
        if i not in classCounter.keys():
            classCounter[i] = 0
        classCounter[i] += 1
    classCounter = sorted(classCounter.items(), reverse=True, key=lambda x: x[-1])
    return classCounter[0][0]

def chooseBestFeatureToSplit(dataset):
    """
    选择最好的划分特征和阈值

    :param dataset (list): 数据集
    :return:
        int: 最佳特征的索引
        float: 最佳阈值
    """
    numFeatures = len(dataset[0]) - 1 
    # baseEntropy = calcShannonEntropy(dataset)
    baseEntropy = calcGini(dataset)
    bestInfoGain = 0
    bestFeature = -1
    bestThreshold = 0
    for i in range(numFeatures):
        featureList = [e[i] for e in dataset]
        uniqueVals = set(featureList)
        uniqueVals = sorted(uniqueVals)
        thresholds = [(uniqueVals[j] + uniqueVals[j + 1]) / 2 for j in range(len(uniqueVals) - 1)]
        for val in thresholds:
            subDataset1, subDataset2 = splitContinuousDataset(dataset, i, val)
            p = len(subDataset1) / float(len(dataset))
            # newEntropy = p * calcShannonEntropy(subDataset1) + (1 - p) * calcShannonEntropy(subDataset2)
            newEntropy = p * calcGini(subDataset1) + (1 - p) * calcGini(subDataset2)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
                bestThreshold = val
    return bestFeature, bestThreshold

def calcGini(dataset):
    num = len(dataset)
    labelCounter = {}
    for data in dataset:
        label = data[-1]
        if label not in labelCounter.keys():
            labelCounter[label] = 0
        labelCounter[label] += 1
    gini = 1
    for key in labelCounter:
        p = labelCounter[key] / float(num)
        gini -= p ** 2
    return gini

def calcShannonEntropy(dataset):
    """
    计算数据集的香农熵

    :param dataset (list): 数据集
    :return
            float: 香农熵
    """
    num = len(dataset)
    labelCounter = {}
    for data in dataset:
        label = data[-1]
        if label not in labelCounter.keys():
            labelCounter[label] = 0
        labelCounter[label] += 1
    shannonEntropy = 0
    for key in labelCounter:
        p = labelCounter[key] / float(num)
        shannonEntropy -= p * log(p, 2)
    return shannonEntropy

def splitContinuousDataset(dataset, axis, value):
    """
    划分连续特征的数据集

    :param dataset (list): 数据集
    :param axis (int): 划分特征的索引
    :param value (float): 划分的阈值

    :return:
        list: 划分后的子数据集1
        list: 划分后的子数据集2
    """
    subDataset1 = [sample[:axis] + sample[axis + 1:] for sample in dataset if sample[axis] <= value]
    subDataset2 = [sample[:axis] + sample[axis + 1:] for sample in dataset if sample[axis] > value]
    return subDataset1, subDataset2


def createDataset(filename='data.csv'):
    dataset = []
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            row_data = []
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
    predictions = []
    for data_point in test_data:
        prediction = traverseTree(tree, labels, data_point)
        predictions.append(prediction)
    return predictions

def traverseTree(tree, labels, data_point):
    if not isinstance(tree, dict):
        return tree

    feature_label = list(tree.keys())[0]
    threshold = tree[feature_label]["threshold"]

    feature_index = labels.index(feature_label)
    feature_value = float(data_point[feature_index])

    if feature_value <= threshold:
        branch_key = "<="
    else:
        branch_key = ">"

    return traverseTree(tree[feature_label][branch_key], labels, data_point)


if __name__ == '__main__':
    dataset, labels = createDataset('../dataset/bill_authentication.csv')
    labels_backup = labels[:]
    
    test_data_size = 30
    test_data = dataset[:test_data_size]
    training_data = dataset[test_data_size + 1:]
    tree = createTree(training_data, labels, max_depth=10)
    # plotTree.createPlot(tree)
    
    real = [e[-1] for e in test_data]
    predictions = predict(tree, labels_backup, test_data)
    predictions = [round(i, 2) for i in predictions]
    print(f"real: {real}\n")
    print(f"predictions: {predictions}\n")
    
    plt.scatter([i for i in range(test_data_size)], predictions, color = 'red', s = 10)
    plt.scatter([i for i in range(test_data_size)], real, color = 'blue', s = 10)
    plt.show()
    