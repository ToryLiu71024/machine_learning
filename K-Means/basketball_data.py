from sklearn.cluster import Birch
from  sklearn.cluster import KMeans

X = [
    [0.0888, 0.5885],
    [0.1399, 0.8291],
    [0.0747, 0.4974],
    [0.0983, 0.5772],
    [0.1276, 0.5703],
    [0.1671, 0.5835],
    [0.1906, 0.5276],
    [0.1061, 0.5523],
    [0.2446, 0.4007],
    [0.167, 0.477],
    [0.2485, 0.4313],
    [0.1227, 0.4909],
    [0.124, 0.5668],
    [0.1461, 0.5113],
    [0.2315, 0.3788],
    [0.0494, 0.559],
    [0.1107, 0.4799],
    [0.2521, 0.5735],
    [0.1007, 0.6318],
    [0.1067, 0.4326],
    [0.1956, 0.428],
    [0.1828, 0.4401],
    [0.1627, 0.5581],
    [0.1403, 0.4866],
    [0.1563, 0.5267],
    [0.2681, 0.5439],
    [0.1236, 0.4419],
    [0.13, 0.3998],
    [0.0896, 0.4325],
    [0.2071, 0.4086],
    [0.2244, 0.4624],
    [0.3437, 0.4325],
    [0.1058, 0.4903],
    [0.2326, 0.4802],
    [0.1577, 0.4345],
    [0.2327, 0.4819],
    [0.1256, 0.6244],
    [0.107, 0.3991],
    [0.1343, 0.4414],
    [0.0586, 0.4013],
    [0.2383, 0.3801],
    [0.1006, 0.3498],
    [0.2164, 0.3185],
    [0.1485, 0.3097],
    [0.227, 0.4319],
    [0.1649, 0.3799],
    [0.1188, 0.4091],
    [0.194, 0.3588],
    [0.2495, 0.4727],
    [0.2378, 0.3212],
    [0.1592, 0.3418],
    [0.2069, 0.4285],
    [0.2084, 0.3917],
    [0.0877, 0.5769],
    [0.101, 0.4773],
    [0.0942, 0.4512],
    [0.055, 0.3096],
    [0.1071, 0.3089],
    [0.0728, 0.4573],
    [0.2771, 0.3214],
    [0.0528, 0.5437],
    [0.213, 0.4121],
    [0.1356, 0.2185],
    [0.1043, 0.3313],
    [0.113, 0.3302],
    [0.1477, 0.4677],
    [0.1317, 0.2406],
    [0.2187, 0.3007],
    [0.2127, 0.2471],
    [0.2547, 0.2894],
    [0.1591, 0.3682],
    [0.0898, 0.389],
    [0.2146, 0.512],
    [0.1871, 0.4449],
    [0.1528, 0.4035],
    [0.156, 0.2683],
    [0.2348, 0.2719],
    [0.1623, 0.3408],
    [0.1239, 0.4393],
    [0.2178, 0.3004],
    [0.1608, 0.3503],
    [0.0805, 0.4388],
    [0.1776, 0.2578],
    [0.1668, 0.2989],
    [0.1072, 0.4455],
    [0.1821, 0.3087],
    [0.188, 0.3678],
    [0.1167, 0.3667],
    [0.2617, 0.3189],
    [0.1994, 0.4187],
    [0.1706, 0.5059],
    [0.1554, 0.3195],
    [0.2282, 0.2381],
    [0.1778, 0.2802],
    [0.1863, 0.381],
    [0.1014, 0.1593]
]

print(X)

clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(X)
print('clf:', clf)

import numpy as  np
import matplotlib.pyplot as plt

print('y_pred:', y_pred)
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)

plt.scatter(x, y, c=y_pred, marker='o')
plt.title('Kmeans-Basketball Data')
plt.xlabel('assists_per_minute')
plt.ylabel('points_per_minute')

plt.legend(['A', 'B', 'C'])

plt.show()