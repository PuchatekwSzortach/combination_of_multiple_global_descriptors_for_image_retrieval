"""
Module with analysis logic
"""

import numpy as np
import scipy.spatial.distance


def get_indices_of_k_most_similar_vectors(vectors, k):
    """
    Given 2D matrix of vectors laid out row-wise compute euclidean distances between all vectors pairs,
    and then for each vector return indices of k most similar vectors.
    Distance of vector with itself is set to large value so that its index will only be included
    in results if k is great or equal to number of vectors

    :param vectors: 2D matrix [n x m] of vectors laid out row-wise
    :param k: int, number of indices of most vectors to return for each vector
    :return: 2D matrix [n x k], indices of k most similar vectors for each vector in input matrix.
    """

    # Compute distances between all vectors
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vectors, metric="euclidean"))

    # For each vector set its distances to itself to maximum number, so it doesn't come up in top k results
    distances += (np.max(distances) + 1) * np.eye(distances.shape[0])

    # For each vector return indices of top k elements with smallest distances to it
    return np.argsort(distances, axis=1)[:, :k]
