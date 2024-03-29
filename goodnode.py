import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def checkDim(dim, data):
    if not dim:
        return 0
    if dim >= data.shape[1]:
        return 0
    return dim


def insert(data, dim=None, node=None, k=8):

    dim = checkDim(dim, data)

    if dim == 0:
        cuts = np.median(data, axis=0)
        if hasattr(node, 'father'):
            node.cuts = cuts
    else:
        cuts = node.father.cuts

    if node is None:
        node = Node(data, cuts, dim, None)

    dataLeft = data[data[:, dim] <= node.cuts[node.dim]]
    dataRight = data[data[:, dim] > node.cuts[node.dim]]

    # Solve issue where we only create complete `Nodes`
    if dataLeft.shape[0] < k:
        return node
    if dataRight.shape[0] < k:
        return node

    dimL = checkDim(dim+1, dataLeft)
    if node.left is None:
        node.left = Node(dataLeft, cuts, dimL, node)
        insert(dataLeft, dimL, node.left, k)

    dimR = checkDim(dim+1, dataRight)
    if node.right is None:
        node.right = Node(dataRight, cuts, dimR, node)
        insert(dataRight, dimR, node.right, k)

    return node


class Node:

    def __init__(self, data, cuts, dim, father):
        self.data = data
        self.cuts = cuts
        self.dim = dim
        self.left = None
        self.right = None
        self.label = None
        self.father = father

    def labelLeafs(self, label=0):
        if not self.left and not self.right:
            self.label = label
            return label+1
        if self.left:
            label = self.left.labelLeafs(label)
        if self.right:
            label = self.right.labelLeafs(label)
        return label

    def search(self, data):

        if self.left is None and self.right is None:
            return self.label

        if data[self.dim] <= self.cuts[self.dim]:
            return self.left.search(data)

        if data[self.dim] > self.cuts[self.dim]:
            return self.right.search(data)


class AdaptivePartitioning(BaseEstimator, TransformerMixin):

    def __init__(self, k: int = 8):
        self.k = k
        self.node = None

    def fit(self, X, y=None):
        self.node = insert(X, k=self.k)
        self.node.labelLeafs()
        return self

    def transform(self, X):
        labels = []
        for i in range(X.shape[0]):
            labels.append(self.node.search(X[i]))
        return labels


X, y = make_blobs(n_samples=1000, centers=4, cluster_std=0.35,
                  n_features=2, random_state=0)

ap = AdaptivePartitioning(k=200)
labels = ap.fit_transform(X)

print(np.bincount(labels))

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
