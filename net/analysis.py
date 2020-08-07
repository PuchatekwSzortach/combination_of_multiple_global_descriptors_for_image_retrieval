"""
Module with analysis logic
"""

import numpy as np
import scipy.spatial.distance
import tensorflow as tf
import tqdm


def get_indices_of_k_most_similar_vectors(vectors, k):
    """
    Given 2D matrix of vectors laid out row-wise compute euclidean distances between all vectors pairs,
    and then for each vector return indices of k most similar vectors.
    Distance of vector with itself is set to large value so that its index will only be included
    in results if k is greater or equal to number of vectors

    :param vectors: 2D matrix [n x m] of vectors laid out row-wise
    :param k: int, number of indices of most similar vectors to return for each vector
    :return: 2D matrix [n x k], indices of k most similar vectors for each vector in input matrix.
    """

    # Compute distances between all vectors
    distances = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vectors, metric="euclidean"))

    # For each vector set its distance to itself to number higher than all other distances,
    # so it doesn't come up in top k results
    distances += (np.max(distances) + 1) * np.eye(distances.shape[0])

    # For each vector return indices of top k elements with smallest distances to it
    return np.argsort(distances, axis=1)[:, :k]


def get_recall_at_k_score(vectors, labels, k):
    """
    Given a 2D matrix of vectors, one per row, and labels, compute ratio of vectors for which
    at least one from k most similar vectors has the same label.
    Euclidean distance is used to define similarity

    :param vectors: [m x n] numpy array, one vector per row
    :param labels: 1D array of ints, labels for each vector
    :param k: int, number of most similar vectors to each vector to consider when computing a score
    :return: float, mean recall score across all vectors
    """

    # Get indices of top k matched vectors for each vector
    top_k_indices_matrix = get_indices_of_k_most_similar_vectors(vectors, k)

    # Select their labels
    top_k_labels = labels[top_k_indices_matrix]

    # For each vector check if any of top k matches has same label, return mean across all vectors
    return np.mean(np.any(top_k_labels == labels.reshape(-1, 1), axis=1))


def get_samples_embeddings(data_loader, prediction_model, verbose):
    """
    Given data loader and prediction model, iterate over whole dataset, predict embeddings for all samples,
    and return (embeddings, labels) tuple

    :param data_loader: data loader that yields (images, labels) batches
    :param model: keras.Model instance
    :param verbose: bool, if True then progress bar is shown
    :return: tuple (embeddings, labels), embeddings is a 2D numpy array with one embedding per row,
    labels is a 1D array of ints
    """

    all_embeddings = []
    all_labels = []

    tf_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(data_loader),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, None, None, 3]), tf.TensorShape([None]))
    ).prefetch(32)

    data_iterator = iter(tf_dataset)

    # Iterate over dataset to obtain embeddings and labels
    for _ in tqdm.tqdm(range(len(data_loader)), disable=not verbose):

        images, labels = next(data_iterator)

        embeddings = prediction_model.predict(images)

        all_embeddings.extend(embeddings)
        all_labels.extend(labels)

    return np.array(all_embeddings), np.array(all_labels)
