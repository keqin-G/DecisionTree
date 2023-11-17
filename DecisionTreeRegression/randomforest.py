import csv
import random
import matplotlib.pyplot as plt
from decisiontree_regression import createTree, traverseTree, createDataset

def createRandomForest(dataset, labels, num_trees=10, max_depth=float('inf'), min_samples_split=2):
    """
        :param dataset: 数据集
        :param labels: 特征集
        :param num_trees: 决策树数量
        :param max_depth: 最大深度
        :param min_samples_split: 最小拆分数
        :return: 随机森林
    """
    forest = []
    for _ in range(num_trees):
        # 有放回的随机抽样
        indices = [random.randint(0, len(dataset) - 1) for _ in range(len(dataset))]
        # 根据随机抽样的下标获取样本
        bootstrap_sample = [dataset[i] for i in indices]
        tree = createTree(bootstrap_sample, labels[:], max_depth, min_samples_split)
        forest.append(tree)
    return forest

def predictRandomForest(forest, labels, test_data):
    predictions = []
    for tree in forest:
        prediction = traverseTree(tree, labels, test_data)
        predictions.append(prediction)
    return sum(predictions) / len(predictions)

if __name__ == '__main__':
    dataset, labels = createDataset('HousingData.csv')
    labels_backup = labels[:]
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
    
    num_trees = 20
    forest = createRandomForest(training_data, labels, num_trees, max_depth=10)

    real = [e[-1] for e in test_data]
    predictions = [predictRandomForest(forest, labels_backup, data_point) for data_point in test_data]
    
    mse = sum((real[i] - predictions[i]) ** 2 for i in range(1, test_data_size)) / test_data_size
    
    print(f"Real: {real}\n")
    print(f"Predictions: {predictions}\n")
    print(f'Mean Squared Error: {mse}')

    plt.scatter([i for i in range(test_data_size)], predictions, color='red', s=10)
    plt.scatter([i for i in range(test_data_size)], real, color='blue', s=10)
    plt.show()