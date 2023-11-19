from math import log, inf
import plotTree
import csv


def createDataset(filename='data.csv'):
    dataset = []
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            dataset.append([float(val) if val.isdigit() else val for val in row[1:]])
  
    labels = dataset[0][:-1]
    dataset = dataset[1:]  
    return dataset, labels


def createTree(dataset, labels, feature_labels, max_depth=float(inf)):
    """
    递归构建决策树

    :param dataset (list): 数据集
    :param labels (list): 特征标签列表
    :param feature_labels (list): 存储构建过程中使用的特征标签

    :return
       dict: 决策树
    """
    classList = [e[-1] for e in dataset]
    if classList.count(classList[0]) == len(classList) or max_depth == 0:  # 如果所有样本属于同一类别，则返回该类别
        return classList[0]
    if len(dataset[0]) == 1:
        return majorityCnt(classList)  # 如果所有特征都用于划分，返回样本中类别最多的类别
    bestFeature = chooseBestFeatureToSplit(dataset)
    bestFeatureLabel = labels[bestFeature]
    feature_labels.append(bestFeatureLabel)
    tree = {bestFeatureLabel: {}}
    del labels[bestFeature]
    featureVal = [e[bestFeature] for e in dataset]
    uniqueVals = set(featureVal)
    for val in uniqueVals:
        subLabels = labels[:]
        tree[bestFeatureLabel][val] = createTree(splitDataset(dataset, bestFeature, val), subLabels, feature_labels,
                                                 max_depth - 1)
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
    选择最好的划分特征

    :param dataset (list): 数据集
    :return
            int: 最佳特征的索引
    """
    numFeatures = len(dataset[0]) - 1
    # baseEntropy = calcShannonEntropy(dataset)
    baseEntropy = calcGini(dataset)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [e[i] for e in dataset]
        uniqueVals = set(featureList)
        newEntropy = 0
        for val in uniqueVals:
            subDataset = splitDataset(dataset, i, val)
            p = len(subDataset) / float(len(dataset))
            newEntropy += p * calcGini(subDataset)
            # newEntropy += p * calcShannonEntropy(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def splitDataset(dataset, asix, val):
    """
    划分数据集

    :param dataset (list): 数据集
    :param axis (int): 划分特征的索引
    :param val: 划分的特征值

    :return
        list: 划分后的数据集
    """
    return [featureVec[:asix] + featureVec[asix + 1:] for featureVec in dataset if featureVec[asix] == val]

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


def predict(tree, feature_labels, test_data):
    predictions = []
    for instance in test_data:
        current_tree = tree.copy()
        # 从父节点开始向下遍历
        while isinstance(current_tree, dict):
            feature = list(current_tree.keys())[0]
            feature_index = feature_labels.index(feature)
            value = instance[feature_index]
            if feature not in current_tree:
                break
            if value not in current_tree[feature]:
                break
            current_tree = current_tree[feature][value]
        predictions.append(current_tree)
    return predictions


if __name__ == '__main__':
    dataset, labels = createDataset('watermelon2.0.csv')
    labels_backup = labels[:]
    featureLabels = []
    tree = createTree(dataset, labels, featureLabels, max_depth=3)
    plotTree.createPlot(tree)

    test_data = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
        ["乌黑", '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],
        ["青绿", '蜷缩', '清脆', '稍糊', '平坦', '硬滑'],
        ["浅白", '硬挺', '沉闷', '清晰', '凹陷', '硬滑'],
        ['青绿', '硬挺', '清脆', '模糊', '平坦', '软粘']
    ]

    predictions = predict(tree, labels_backup, test_data)
    print(predictions)

