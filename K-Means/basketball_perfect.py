from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = []
for line in open('basketball_data.txt', 'r').readlines():
    # 去除换行符
    line = line.rstrip()
    # 将中间多余的空格替换为一个空格
    result = ' '.join(line.split())
    # 将数据转化为float类型的数组
    s = [float(x) for x in result.strip().split(' ')]
    data.append(s)
print(data)

# 第一列数据：每分钟助攻数（assists_per_minute）
L1 = [n[0] for n in data]
# 第五列数据：每分钟的分数（points_per_minute）
L5 = [n[4] for n in data]

# 连列数据生成二维数据
T = dict(zip(L1, L5))
print(T)

X = list(map(lambda x, y: (x, y), T.keys(), T.values()))
print(X)

# 聚类，要指定要聚类的堆数
clf = KMeans(n_clusters=3)
# 聚类结果被标记为0，1，2
y_pred = clf.fit_predict(X)

x = [n[0] for n in X]
y = [n[1] for n in X]

x1 = []
y1 = []

x2 = []
y2 = []

x3 = []
y3 = []

for i in range(len(X)):
    if y_pred[i] == 0:
        x1.append(X[i][0])
        y1.append(X[i][1])
    elif y_pred[i] == 1:
        x2.append(X[i][0])
        y2.append(X[i][1])
    elif y_pred[i] == 2:
        x3.append(X[i][0])
        y3.append(X[i][1])

plot1, = plt.plot(x1, y1, 'or', marker='x')
plot2, = plt.plot(x2, y2, 'og', marker='o')
plot3, = plt.plot(x3, y3, 'ob', marker='*')

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend((plot1, plot2, plot3), ('A', 'B', 'C'), fontsize=10)

plt.show()
