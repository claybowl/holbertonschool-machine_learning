#!/usr/bin/env python3
"""12-agglomerative
performs K-means on a dataset
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset"""
    Z = scipy.cluster.hierarchy.linkage(X, 'ward')

    dn = scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist,
                                            above_threshold_color='b')
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, dist, 'distance')
    return clss
