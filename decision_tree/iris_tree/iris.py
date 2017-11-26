from sklearn.datasets import load_iris  # 鸢尾花数据集
from sklearn.tree import DecisionTreeClassifier  # 决策树DTC包
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
print(iris)
print(iris.target)
print(iris.data.shape)

# 训练
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
print(clf)

# 预测
predicted = clf.predict(iris.data)

# 画图
X = iris.data

L1 = [x[0] for x in X]
L2 = [x[1] for x in X]

plt.scatter(L1, L2, c=predicted, marker='x')
plt.title("DTC")
plt.show()
