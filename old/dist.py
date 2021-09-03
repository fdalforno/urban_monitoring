from scipy.spatial import distance as dist
import numpy as np


np.random.seed(42)

objectCentroids = np.random.uniform(size=(2, 2))
centroids = np.random.uniform(size=(3, 2))

print(objectCentroids)
print(centroids)

D = dist.cdist(objectCentroids, centroids)
print(D)

rows = D.min(axis=1).argsort()
print(rows)

cols = D.argmin(axis=1)[rows]
print(cols)

print(list(zip(rows, cols)))