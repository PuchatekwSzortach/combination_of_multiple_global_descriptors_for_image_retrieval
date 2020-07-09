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
            outputs=[self.output]
        )

        self.model.compile(
            optimizer='adam',
            loss=get_batch_hard_triplets_loss_op
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


def get_batch_hard_triplets_loss_op(labels, embeddings):
    """
    Implementation of batch-hard triplets loss from
    "In Defense of the Triplet Loss for Person Re-Identification" paper

    :param labels: 1D tensor, batch of labels for embeddings
    :param embeddings: 2D tensor, batch of embeddings

    :return: loss tensor
    """

    # Keras adds an unnecessary batch dimension on our labels, flatten them
    flat_labels = tf.reshape(labels, shape=(-1,))

    distances_matrix_op = get_distances_matrix_op(embeddings)
    positives_mask = get_vector_elements_equalities_matrix_op(flat_labels)

    # For each anchor, select largest distance to same category element
    hard_positives_vector_op = tf.reduce_max(distances_matrix_op * positives_mask, axis=1)

    max_distance_op = tf.reduce_max(distances_matrix_op)

    # Modifie distances matrix so that all distances between positive pairs are set higher than all
    # distances between negative pairs
    distances_matrix_op_with_positive_distances_maxed_out = distances_matrix_op + (positives_mask * max_distance_op)

    hard_negatives_vector_op = tf.reduce_min(distances_matrix_op_with_positive_distances_maxed_out, axis=1)

    losses_vector_op = tf.nn.relu(1.0 + hard_positives_vector_op - hard_negatives_vector_op)

    return tf.reduce_mean(losses_vector_op)


def get_distances_matrix_op(matrix_op):
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

    rows_count_op = tf.shape(matrix_op)[0]

    repeated_rows_inputs = tf.repeat(matrix_op, repeats=rows_count_op, axis=0)
    repeated_matrix_inputs = tf.tile(matrix_op, multiples=(rows_count_op, 1))

    differences = repeated_rows_inputs - repeated_matrix_inputs

    # Result is a 1D vector of distances
    distances_vector_op = tf.norm(differences, axis=1)

    # So reshape it to a matrix of same shape as input
    return tf.reshape(
        tensor=distances_vector_op,
        shape=(rows_count_op, rows_count_op)
    )


def get_vector_elements_equalities_matrix_op(vector_op):
    """
    Given a vector_op, return a square matrix such that element (i,j) is 1 if
    vector_op[i] == vector_op[j] and 0 otherwise

    :param vector_op: 1D tensor of ints
    :return: 2D matrix of ints
    """

    elements_count_op = tf.shape(vector_op)[0]

    # Unroll vector so that each element can be matched with each other element
    vector_repeated_elements_wise = tf.repeat(vector_op, repeats=elements_count_op)
    vector_repeated_vector_wise = tf.tile(vector_op, multiples=[elements_count_op])

    # Compute equalities, cast booleans to floats
    equalities_vector_op = tf.cast(vector_repeated_elements_wise == vector_repeated_vector_wise, tf.float32)

    # Reshape vector to square matrix
    return tf.reshape(equalities_vector_op, shape=(elements_count_op, elements_count_op))
