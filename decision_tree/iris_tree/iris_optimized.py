from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

# 将数分为训练数据以及测试数据
train_data = np.concatenate((iris.data[0:40, :], iris.data[50:90, :], iris.data[100:140, :]), axis=0)
train_target = np.concatenate((iris.target[0:40], iris.target[50:90], iris.target[100:140]), axis=0)
test_data = np.concatenate((iris.data[40:50, :], iris.data[90:100, :], iris.data[140:150, :]), axis=0)
test_target = np.concatenate((iris.target[40:50], iris.target[90:100], iris.target[140:150]), axis=0)

# 训练
clf = DecisionTreeClassifier()
clf.fit(train_data, train_target)

# 预测
predict_target = clf.predict(test_data)
print('预测结果', predict_target)

print('预测与原值对比', sum(predict_target == test_target))
# 输出准确率 召回率 F值
print(metrics.classification_report(test_target, predict_target))
print(metrics.confusion_matrix(test_target, predict_target))

# 绘图
X = test_data
L1 = [n[0] for n in X]
L2 = [n[1] for n in X]
plt.scatter(L1, L2, c=predict_target, marker='x')
plt.title("DecisionTreeClassifier")
plt.show()
