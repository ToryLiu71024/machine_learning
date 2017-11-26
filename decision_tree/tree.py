from math import log
import operator


def calcShannonEnt(dataSet):
    '''计算shannon熵'''
    numEntries = len(dataSet)  # 数据集长度
    labelCounts = {}
    # 遍历数据集
    for featVec in dataSet:
        # 最后一列的值
        currentLabel = featVec[-1]
        # 如果当前值不在字典的键中
        if currentLabel not in labelCounts.keys():
            # 将当前值加入字典
            labelCounts[currentLabel] = 0
            # 每个键的值记录当前类别出现的次数
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 类别出现的概率
        prob = float(labelCounts[key]) / numEntries
        # 计算shannon熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    '''将数据集分组
        dataSet  待划分的数据集
        axis  划分数据集的特征
        value  特征的返回值
    '''
    retDataSet = []
    # 遍历数据集
    for featVec in dataSet:
        # 发现符合要求的值
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''选择最好的分组方式'''
    numFeatures = len(dataSet[0]) - 1  # 数据集的特征个数
    baseEntropy = calcShannonEnt(dataSet)  # 整个数据集的原始shannon熵
    bestInfoGain = 0.0
    bestFeature = -1
    # 遍历数据集中的所有特征
    for i in range(numFeatures):
        # 将数据集中所有第i个特征值或者可能存在的值放入featList
        featList = [example[i] for example in dataSet]
        # 对featList去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值
        for value in uniqueVals:
            # 对每个特征划分一次数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 当前划分的shannon熵
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算当前特征的信息增益
        infoGain = baseEntropy - newEntropy
        # 比较所有特征中的信息增益，返回最好特征划分的索引值
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # 返回索引值
    return bestFeature


def majorityCnt(classList):
    '''得到出现次数最多的分类名称'''
    classCount = {}
    # 遍历列表
    for vote in classList:
        # 同代码14-18行
        # 主要作用是对列表中的元素进行计数
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    # 排序操作
    sortedClassCount = sorted(classCount.items(),  # 要排序的对象
                              key=operator.itemgetter(1),  # 按照第1个域排序
                              reverse=True)  # 降序排列
    # 返回出现次数最多的分类名称
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''
    创建树
    
    dataSet:数据集
    labels：标签列表
    '''
    # 包含了数据集的所有类标签
    classList = [example[-1] for example in dataSet]
    # 如果所有类标签完全相同，直接返回该类标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果用完了所有的特征，仍不能将数据集划分为仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        # 出现次数最多的类别作为返回值
        return majorityCnt(classList)
    # 最好的分组方式的索引值
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 得到最好分组方式的类标签
    bestFeatLabel = labels[bestFeat]
    # 声明树字典
    myTree = {bestFeatLabel: {}}
    # 删除已完成分类使用的类标签
    del (labels[bestFeat])
    # 生成按照分组区分的属性值列表
    featValues = [example[bestFeat] for example in dataSet]
    # 去重
    uniqueValues = set(featValues)
    # 遍历当前选择特征包含的所有属性值
    for value in uniqueValues:
        # 复制了类标签
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
                                                               bestFeat,
                                                               value),
                                                  subLabels)
    # 返回树结构
    return myTree


myDat, labels = createDataSet()
newLabels = labels.copy()
myTree = createTree(myDat, labels)
print(myTree)


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = []
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


print(classify(myTree, newLabels, [1, 0]))
