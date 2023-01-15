import pytest
import numpy as np
from .adaptive_partitioning import AdaptivePartitioning
from sklearn.datasets import make_blobs


def test_basic_clustering():
    # Checks if four points give four clusters
    # given k = 1
    X = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    est = AdaptivePartitioning(k=1)
    labels = est.fit_transform(X)
    assert np.unique(labels).shape[0] == 4


@pytest.mark.parametrize("niter", [None, 1, 2, 3, 4])
def test_klimit(niter):
    # Checks if four points give four clusters
    # given k = 2
    X = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                 [-1, -1], [1, -1], [1, 1], [-1, 1]])
    est = AdaptivePartitioning(k=2, niter=niter)
    labels = est.fit_transform(X)
    assert np.unique(labels).shape[0] == 4


@pytest.mark.parametrize("niter", [None, 1, 2, 3, 4])
def test_same_labels(niter):
    X = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                  [-2, -2], [2, -2], [2, 2], [-2, 2]])
    est = AdaptivePartitioning(k=2, niter=niter)
    est.fit(X)
    labels = est.transform(np.array([[-1, -1], [-2, -2]]))
    assert np.unique(labels).shape[0] == 1


@pytest.mark.parametrize("niter", [None, 1, 2, 3, 4])
def test_different_labels(niter):
    X = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                  [-2, -2], [2, -2], [2, 2], [-2, 2]])
    est = AdaptivePartitioning(k=2, niter=niter)
    est.fit(X)
    labels = est.transform(np.array([[-1, -1], [2, 2]]))
    assert np.unique(labels).shape[0] == 2


@pytest.mark.parametrize("niter", [None, 1, 2, 3, 4])
def test_k_min(niter):
    # Check if K really limits the data in each terminal node
    X, y = make_blobs(n_samples=1000, centers=4,
                      cluster_std=0.35, n_features=4, random_state=0)
    est = AdaptivePartitioning(k=50, niter=niter)
    labels = est.fit_transform(X)
    assert np.bincount(labels).min() >= 50


def recursivecheck(node, featureNumber):
    resp = False
    if hasattr(node, 'label') and node.dim == featureNumber:
        return True
    else:
        if node.left:
            resp = recursivecheck(node.left, featureNumber)
        if node.right:
            resp = recursivecheck(node.right, featureNumber)
    return resp


@pytest.mark.parametrize("featureNumber", [0, 1, 2, 3])
def test_variable_usage(featureNumber):
    # Checks if the variables were used
    X, y = make_blobs(n_samples=10000, centers=4,
                      cluster_std=0.35, n_features=4, random_state=0)
    est = AdaptivePartitioning(k=1)
    est.fit(X)
    assert recursivecheck(est.node_, featureNumber)


@pytest.mark.parametrize("niter", [2, 3, 4])
def test_niter_effect(niter):
    # Checks if the variables were used
    X, y = make_blobs(n_samples=10000, centers=4,
                      cluster_std=0.35, n_features=4, random_state=0)

    niter_less = niter-1

    est = AdaptivePartitioning(k=1, niter=niter_less)
    labels_less = est.fit_transform(X)

    est = AdaptivePartitioning(k=1, niter=niter)
    labels = est.fit_transform(X)

    assert np.unique(labels).shape[0] > np.unique(labels_less).shape[0]


def test_accept_list():
    X_ = [[-1, -1], [1, -1], [1, 1], [-1, 1],
          [-2, -2], [2, -2], [2, 2], [-2, 2]]
    X = np.array(X_)

    est1 = AdaptivePartitioning(k=2)
    labels1 = est1.fit_transform(X)

    est2 = AdaptivePartitioning(k=2)
    labels2 = est2.fit_transform(X_)

    assert np.bincount(labels1).shape[0] == np.bincount(labels2).shape[0]
