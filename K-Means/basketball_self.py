from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = []

for line in open('basketball_data.txt', 'r').readlines():
    line = line.rstrip()
    result = ' '.join(line.split())
    s = [float(x) for x in result.split(' ')]
    data.append(s)

print(data)

L1 = [n[0] for n in data]
L2 = [n[4] for n in data]
print(L1)
print(L2)

T = dict(zip(L1, L2))
# X =[[ , ],[ , ]]
X = list(map(lambda x, y: (x, y), T.keys(), T.values()))
print(X)

clf = KMeans(n_clusters=3)
y_prop = clf.fit_predict(X)

x1 = []
y1 = []

x2 = []
y2 = []

x3 = []
y3 = []

for i in range(len(X)):
    if y_prop[i] == 0:
        x1.append(X[i][0])
        y1.append(X[i][1])
    elif y_prop[i] == 1:
        x2.append(X[i][0])
        y2.append(X[i][1])
    elif y_prop[i] == 2:
        x3.append(X[i][0])
        y3.append(X[i][1])

plot1, = plt.plot(x1, y1, 'or', marker='x')
plot2, = plt.plot(x2, y2, 'og', marker='o')
plot3, = plt.plot(x3, y3, 'ob', marker='*')

plt.title("K-Means Basketball Data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
plt.legend((plot1, plot2, plot3), ('A', 'B', 'C'), fontsize=15)

plt.show()
