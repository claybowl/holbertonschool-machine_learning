#!/usr/bin/env python3
"""12-agglomerative
performs K-means on a dataset
"""


def agglomerative(X, dist):
    # Import necessary functions
    import scipy.cluster.hierarchy as sch
    import matplotlib.pyplot as plt

    # Perform agglomerative clustering
    Z = sch.linkage(X, method='ward')

    # Plot dendrogram
    plt.figure(figsize=(25, 10))
    dn = sch.dendrogram(Z, color_threshold=dist)
    plt.show()

    # Get cluster labels
    clss = sch.fcluster(Z, dist, criterion='distance')

    return clss
