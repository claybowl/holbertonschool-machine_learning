#!/usr/bin/env python3
"""10-kmeans
performs K-means on a dataset
"""


def kmeans(X, k):
    # Import necessary functions
    from sklearn.cluster import KMeans

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k).fit(X)

    # Get cluster centroids and labels
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
