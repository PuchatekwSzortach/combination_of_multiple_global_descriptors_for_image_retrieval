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

        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation=tf.nn.swish)(base_model.output)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation=tf.nn.swish)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1024, activation=tf.nn.swish)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units=128, activation=None)(x)

        self.output = x

        self.model = tf.keras.models.Model(
            inputs=self.input,
            outputs=[self.output]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=get_hard_aware_point_to_set_loss_op
        )


class HardAwarePointToSetLossBuilder:
    """
    A helper class for building hard aware point to set loss ops
    """

    @staticmethod
    def get_weighted_losses_vector_op(distances_matrix_op, mask_op, polynomial_power):
        """
        Get losses vector op from query vectors to sets specified by mask op.

        :param distances_matrix_op: 2D tensor op with distances from queries to images in batch,
        each row represents distances from one query to all images in a batch
        :param mask_op: 2D tensor with 1s for elements that should be used in computations and 0s for elements
        that should be masked
        :param polynomial_power: float, power used for computing per sample weights
        :return: 1D tensor of weighted point to scale distances, each element represents weighted sum of distances
        between a query and all the non-masked elements from image set
        """

        # Compute weights, after computing multiply by mask, so that any elements that shouldn't be included
        # in computations have their weights zeroed out

        tf.print("\nInside HardAwarePointToSetLossBuilder.get_weighted_losses_vector_op")

        weights_op = tf.math.pow(distances_matrix_op + 1.0, polynomial_power) * mask_op

        weighted_distances_op = distances_matrix_op * weights_op

        tf.print("\nweights sum op")
        tf.print(tf.math.reduce_sum(weights_op, axis=1))

        normalized_weighted_point_to_set_distances = \
            tf.math.reduce_sum(weighted_distances_op, axis=1) / tf.math.reduce_sum(weights_op, axis=1)

        return normalized_weighted_point_to_set_distances


@tf.function
def get_hard_aware_point_to_set_loss_op(labels, embeddings):
    """
    Implementation of loss from
    "Hard-aware point-to-set deep metric for person re-identification" paper

    :param labels: 1D tensor, batch of labels for embeddings
    :param embeddings: 2D tensor, batch of embeddings
    :return: loss tensor
    """

    tf.print("\nembeddings")
    tf.print(embeddings)

    if tf.math.reduce_max(tf.cast(tf.math.is_nan(embeddings), tf.float32)) > 0:
        tf.print("\n\nNaN embeddings detected!")
    else:
        tf.print("\n\nEmbeddings are fine")

    # Keras adds an unnecessary batch dimension on our labels, flatten them
    flat_labels = tf.reshape(labels, shape=(-1,))

    distances_matrix_op = get_distances_matrix_op(embeddings)

    tf.print("\ndistances_matrix_op is")
    tf.print(distances_matrix_op)

    positives_mask = get_vector_elements_equalities_matrix_op(flat_labels)

    tf.print("\npositives_mask is")
    tf.print(positives_mask)

    alpha_power_factor = 10.0

    hard_positives_vector_op = HardAwarePointToSetLossBuilder.get_weighted_losses_vector_op(
        distances_matrix_op=distances_matrix_op,
        mask_op=positives_mask,
        polynomial_power=alpha_power_factor
    )

    tf.print("\nhard_positives_vector_op is")
    tf.print(hard_positives_vector_op)

    hard_negatives_vector_op = HardAwarePointToSetLossBuilder.get_weighted_losses_vector_op(
        distances_matrix_op=distances_matrix_op,
        mask_op=1.0 - positives_mask,
        polynomial_power=-2.0 * alpha_power_factor
    )

    tf.print("\nhard_negatives_vector_op is")
    tf.print(hard_negatives_vector_op)

    # Use soft margin loss instead of hinge loss, as per "In defence of the triplet loss" paper
    losses_vector_op = tf.math.log1p(tf.math.exp(hard_positives_vector_op - hard_negatives_vector_op))

    loss_op = tf.reduce_mean(losses_vector_op)

    tf.print("\n\nLoss op is")
    tf.print(loss_op)

    return loss_op


@tf.function
def get_batch_hard_triplets_loss_op(labels, embeddings):
    """
    Implementation of batch-hard triplets loss from
    "In Defense of the Triplet Loss for Person Re-Identification" paper

    :param labels: 1D tensor, batch of labels for embeddings
    :param embeddings: 2D tensor, batch of embeddings

    :return: loss tensor
    """

    if tf.math.reduce_max(tf.cast(tf.math.is_nan(embeddings), tf.float32)) > 0:
        tf.print("\n\nNaN embeddings detected")
    else:
        tf.print("\n\nEmbeddings are fine")

    # Keras adds an unnecessary batch dimension on our labels, flatten them
    flat_labels = tf.reshape(labels, shape=(-1,))

    distances_matrix_op = get_distances_matrix_op(embeddings)

    positives_mask = get_vector_elements_equalities_matrix_op(flat_labels)

    # For each anchor, select largest distance to same category element
    hard_positives_vector_op = tf.reduce_max(distances_matrix_op * positives_mask, axis=1)

    max_distance_op = tf.reduce_max(distances_matrix_op)

    # Modify distances matrix so that all distances between positive pairs are set higher than all
    # distances between negative pairs
    distances_matrix_op_with_positive_distances_maxed_out = distances_matrix_op + (positives_mask * max_distance_op)

    hard_negatives_vector_op = tf.reduce_min(distances_matrix_op_with_positive_distances_maxed_out, axis=1)

    # Use soft margin loss instead of hinge loss, as per "In defence of the triplet loss" paper
    losses_vector_op = tf.math.log1p(tf.math.exp(hard_positives_vector_op - hard_negatives_vector_op))

    loss = tf.reduce_mean(losses_vector_op)

    return loss


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
