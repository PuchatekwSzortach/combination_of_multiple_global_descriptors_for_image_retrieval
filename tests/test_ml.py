"""
Tests for net.ml module
"""

import numpy as np
import tensorflow as tf

import net.ml


def test_get_distance_matrix_op():
    """
    Test computing a matrix of distances between all rows permutations of a matrix.
    """

    inputs_matrix = np.array([
        [1, 3, 5, 7],
        [2, 2, 4, 4],
        [1.5, -2, -4, 0]
    ], dtype=np.float32)

    expected = np.sqrt(np.array([
        [0, 12, 155.25],
        [12, 0, 96.25],
        [155.25, 96.25, 0]
    ]))

    actual = net.ml.get_distances_matrix_op(
        matrix_op=tf.constant(inputs_matrix)
    ).numpy()

    assert np.all(np.isclose(expected, actual))


def test_get_vector_elements_equalities_matrix_op():
    """
    Test net.get_vector_elements_equalities_matrix_op
    """

    vector = np.array([1, 4, 3, 1, 4, 3])

    expected = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1]
    ])

    actual = net.ml.get_vector_elements_equalities_matrix_op(vector_op=tf.constant(vector)).numpy()

    assert np.all(expected == actual)


def test_average_ranking_position():
    """
    Test average_ranking_position function
    """

    labels = tf.reshape(tf.constant([1, 2, 3, 1, 2], dtype=tf.float32), (-1, 1))

    embeddings = tf.constant([
        [1, 1],
        [2, 2],
        [3, 3],
        [2.5, 2.5],
        [10, 10]
    ], dtype=tf.float32)

    expected = np.mean([1, 2, 0, 1.5, 1.5])

    actual = net.ml.average_ranking_position(
        labels=labels,
        embeddings=embeddings
    )

    assert np.isclose(actual, expected)
