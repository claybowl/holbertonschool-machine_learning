#!/usr/bin/env python3
"""12-agglomerative
performs K-means on a dataset
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on the given
    data and plots a dendrogram.

    Parameters:
    X (numpy.ndarray): The data to be clustered. Shape (n, d),
    where n is the number of data points and d is the number of dimensions.
    dist (float): The distance threshold for the formation of clusters.

    Returns:
    numpy.ndarray: An array of shape (n,) containing the
    cluster labels for each data point.
    """
    # Perform agglomerative clustering
    linkage_matrix = scipy.cluster.hierarchy.linkage(X, 'ward')

    # Plot dendrogram
    scipy.cluster.hierarchy.dendrogram(
        linkage_matrix, color_threshold=dist, above_threshold_color='b')
    plt.show()

    # Assign each data point to a cluster
    labels = scipy.cluster.hierarchy.fcluster(linkage_matrix, dist, 'distance')

    return labels
