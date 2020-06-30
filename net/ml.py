"""
Module with machine learning code
"""

import tensorflow as tf


class ImagesSimilarityComputer:
    """
    Class for computing similarity between images.
    """

    def __init__(self):
        """
        Constructor
        """

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(256, 256, 3)
        )

        self.input = base_model.input

        x = tf.keras.layers.Flatten()(base_model.output)
        x = tf.keras.layers.Dense(units=2048, activation=tf.nn.swish)(x)
        x = tf.keras.layers.Dense(units=1048, activation=None)(x)

        self.output = x

        self.model = tf.keras.models.Model(
            inputs=self.input,
            outputs=self.output
        )


def get_hard_aware_point_to_set_loss_op(embeddings, labels):
    """
    Implementation of loss from
    "Hard-aware point-to-set deep metric for person re-identification" paper

    :param embeddings: 2D tensor, batch of embeddings
    :param labels: 1D tensor, batch of labels for embeddings
    :return: loss tensor
    """

    raise NotImplementedError()


def get_batch_hard_triplets_loss_op(embeddings, labels):
    """
    Implementation of batch-hard triplets loss from
    "In Defense of the Triplet Loss for Person Re-Identification" paper

    :param embeddings: 2D tensor, batch of embeddings
    :param labels: 1D tensor, batch of labels for embeddings
    :return: loss tensor
    """

    raise NotImplementedError()


def get_distance_matrix_op(matrix_op):
    """
    Given a 2D matrix tensor, return euclidean distance between each row combination

    :param matrix_op: 2D tensor
    :type matrix_op: 2D tensor
    """

    # Unroll matrix so that each row can be matched with each other row
    # repeated_rows_inputs repeats each row n times in order
    # (first row n times, then second row n times, then third, etc)
    # repeated_matrix_inputs repaets each row n times in order
    # whole matrix first time, whole matrix second time, etc
    # This way we will have two matrices such that all rows combinations can be matched
    repeated_rows_inputs = tf.repeat(matrix_op, repeats=matrix_op.shape[0], axis=0)
    repeated_matrix_inputs = tf.tile(matrix_op, multiples=(matrix_op.shape[0], 1))

    differences = repeated_rows_inputs - repeated_matrix_inputs

    # Result is a 1D vector of distances
    distances_vector_op = tf.norm(differences, axis=1)

    # So reshape it to a matrix of same shape as input
    return tf.reshape(
        tensor=distances_vector_op,
        shape=(matrix_op.shape[0], matrix_op.shape[0])
    )
