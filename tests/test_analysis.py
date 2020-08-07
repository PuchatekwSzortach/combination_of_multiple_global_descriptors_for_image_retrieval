"""
Tests for analysis module
"""

import numpy as np

import net.analysis


def test_get_indices_of_k_most_similar_vectors():
    """
    Test get_indices_of_k_most_similar_vectors
    """

    vectors = np.array([
        [1, 1],
        [10, 10],
        [2.5, 2.5],
        [5, 5],
        [8, 8]
    ])

    actual = net.analysis.get_indices_of_k_most_similar_vectors(
        vectors=vectors,
        k=3
    )

    expected = np.array([
        [2, 3, 4],
        [4, 3, 2],
        [0, 3, 4],
        [2, 4, 0],
        [1, 3, 2]
    ])

    assert np.all(expected == actual)
