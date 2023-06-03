#!/usr/bin/env python3
"""11-gmm
performs K-means on a dataset
"""
import sklearn.mixture


def gmm(X, k):
    """
    Performs Gaussian Mixture Model (GMM) clustering on the given data.

    Parameters:
    X (numpy.ndarray): The data to be clustered. Shape (n, d),
    where n is the number of data points and d is the number of dimensions.
    k (int): The number of clusters.

    Returns:
    tuple: A tuple containing the weights, means, and
    covariances of the Gaussian components,
           the labels for each data point, and the
           Bayesian Information Criterion (BIC).
    """
    # Perform GMM clustering
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k)

    # Fit the model to the data
    gmm_model.fit(X)

    # Get cluster parameters and labels
    weights = gmm_model.weights_
    means = gmm_model.means_
    covariances = gmm_model.covariances_
    labels = gmm_model.predict(X)

    # Calculate BIC
    bic = gmm_model.bic(X)

    return weights, means, covariances, labels, bic
