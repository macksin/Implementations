from adaptive_partitioning import AdaptivePartitioning
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

X, y = make_blobs(n_samples=1000, centers=4, cluster_std=0.35,
                  n_features=2, random_state=0)

# Transform
adaptivePartitioning = AdaptivePartitioning(k=200)
labels = adaptivePartitioning.fit_transform(X)

print(np.bincount(labels))

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
