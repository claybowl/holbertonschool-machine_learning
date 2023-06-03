#!/usr/bin/env python3
"""10-kmeans
performs K-means on a dataset
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on the given data.

    Parameters:
    X (numpy.ndarray): The data to be clustered.
    Shape (n, d), where n is the number of data points and d is the number of dimensions.
    k (int): The number of clusters.

    Returns:
    tuple: A tuple containing the cluster centroids
    and the labels for each data point.
    """
    # Perform K-means clustering
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)

    # Fit the model to the data
    kmeans_model.fit(X)

    # Get cluster centroids and labels
    centroids = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_

    return centroids, labels
