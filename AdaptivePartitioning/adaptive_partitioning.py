import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


def checkDim(dim, data):
    if not dim:
        return 0
    if dim >= data.shape[1]:
        return 0
    return dim


def insert(data, dim=None, node=None, k=8, niter=None, iter=None):

    if not iter:
        iter = 0

    dim = checkDim(dim, data)

    if dim == 0:
        cuts = np.median(data, axis=0)
        if hasattr(node, 'father'):
            node.cuts = cuts
            iter += 1
    else:
        cuts = node.father.cuts

    if node is None:
        node = Node(data, cuts, dim, None)

    if niter and iter:
        if niter <= iter:
            return node

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
        insert(dataLeft, dimL, node.left, k, niter, iter)

    dimR = checkDim(dim+1, dataRight)
    if node.right is None:
        node.right = Node(dataRight, cuts, dimR, node)
        insert(dataRight, dimR, node.right, k, niter, iter)

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

    def __init__(self, k: int = 8, niter=None):
        self.k = k
        self.niter = niter

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse='csr')
        self.node_ = insert(X, k=self.k, niter=self.niter)
        self.node_.labelLeafs()
        return self

    def transform(self, X):
        check_is_fitted(self, attributes="node_")

        X = check_array(X, accept_sparse='csr')

        labels = []
        for i in range(X.shape[0]):
            labels.append(self.node_.search(X[i]))
        return labels
