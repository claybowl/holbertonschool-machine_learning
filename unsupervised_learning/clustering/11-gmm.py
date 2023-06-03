#!/usr/bin/env python3
"""11-gmm
performs K-means on a dataset
"""
import sklearn.cluster


def gmm(X, k):
    """performs k-means on a dataset"""
    # Perform GMM clustering
    gmm = GaussianMixture(n_components=k).fit(X)

    # Get cluster parameters and labels
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, m, S, clss, bic
