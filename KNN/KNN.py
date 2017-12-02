from numpy import *  # 科学计算包
import operator  # 运算符模块

'''
MachineLearningAction

2.Classifying with k-Nearest Neighbors
'''


def createDataSet():
    '''
    创建数据集
    :return: 
    '''
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()
print("group:%s,\n labels%s" % (group, labels))


def classify0(inX, dataSet, labels, k):
    '''
    计算距离
    :param inX: 用于分类的输入向量。即将对其进行分类。
    :param dataSet: 训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻居的数目
    :return: 
    '''
    dataSetSize = dataSet.shape[0]  # 得到数组的行数。即知道有几个训练数据
    # tile:numpy中的函数。
    # tile将原来的一个数组，扩充成了4个一样的数组。
    # diffMat得到了目标与训练数值之间的差值。
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2  # 各个元素分别平方
    sqDistances = sqDiffMat.sum(axis=1)  # 对应列相乘，即得到了每一个距离的平方
    distances = sqDistances ** 0.5  # 开方，得到距离。
    sortedDistIndicies = distances.argsort()  # 升序排列
    # 选择距离最小的k个点。
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


print(classify0([0, 0], group, labels, 3))


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros(numberOfLines, 3)
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFormLine = line.strip('\t')
        returnMat[index, :] = listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index += 1
    return returnMat, classLabelVector
