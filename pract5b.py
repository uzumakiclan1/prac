#Radial Basis function

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X = np.random.rand(100, 2)
y = np.sin(X[:, 0]) + np.cos(X[:, 1])

def rbf(x, center, sigma=0.5):
    return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

kmeans = KMeans(n_clusters=10).fit(X)
centers = kmeans.cluster_centers_

R = np.array([[rbf(x, center) for center in centers] for x in X])
W = np.dot(np.linalg.pinv(R), y)

def rbf_network(X, centers, W, sigma=0.5):
    R = np.array([[rbf(x, center, sigma) for center in centers] for x in X])
    return np.dot(R, W)

y_pred = rbf_network(X, centers, W)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")
