"""
Module with machine learning code
"""

import tensorflow as tf


class ImagesSimilarityComputer:
    """
    Class for computing similarity between images.
    """

    def __init__(self, image_size):
        """
        Constructor

        :param image_size: int, height and width of input image for the model
        """

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3)
        )

        self.input = base_model.input

        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation=tf.nn.swish)(base_model.output)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation=tf.nn.swish)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1024, activation=tf.nn.swish)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units=128, activation=None)(x)

        self.output = tf.math.l2_normalize(x, axis=1)

        self.model = tf.keras.models.Model(
            inputs=self.input,
            outputs=[self.output]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=get_hard_aware_point_to_set_loss_op
        )


class CGDImagesSimilarityComputer:
    """
    Class for computing similarity between images based on Combination of Multiple Global Descriptors model
    """

    def __init__(self, image_size):
        """
        Constructor

        :param image_size: int, height and width of input image for the model
        """

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3)
        )

        self.input = base_model.input

        x = base_model.output

        batch_of_channels_norms = self._get_batch_of_channels_norms(x)

        sum_of_pooling_convolutions_features = self._get_normalized_branch(
            x=self._get_sum_of_pooling_convolutions_head(x, batch_of_channels_norms),
            target_size=512)

        maximum_activations_of_convolutions_features = self._get_normalized_branch(
            x=self._get_maximum_activation_of_convolutions_head(x, batch_of_channels_norms),
            target_size=512)

        generalized_mean_pooling_features = self._get_normalized_branch(
            x=self._get_generalized_mean_pooling_head(x, batch_of_channels_norms),
            target_size=512)

        combination_of_multiple_global_descriptors = tf.math.l2_normalize(
            tf.concat([
                sum_of_pooling_convolutions_features,
                maximum_activations_of_convolutions_features,
                generalized_mean_pooling_features], axis=1),
            axis=1)

        self.output = combination_of_multiple_global_descriptors

        self.model = tf.keras.models.Model(
            inputs=self.input,
            outputs=[self.output]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=get_hard_aware_point_to_set_loss_op
        )

    @staticmethod
    def _get_batch_of_channels_norms(x):

        # Square all elements
        squared_tensor = tf.math.square(x)

        # Compute sums of squared elements across channels.
        # Output is a 2D matrix, one row per sample.
        # One element in a row represents norm of a single channel for a sample.
        batch_of_sums_of_squared_channels = tf.reduce_sum(squared_tensor, axis=(1, 2))

        epsilon = 1e-6

        return tf.math.sqrt(batch_of_sums_of_squared_channels) + epsilon

    @staticmethod
    def _get_normalized_branch(x, target_size):

        x = tf.keras.layers.Dense(units=target_size, activation=tf.nn.swish)(x)
        return tf.math.l2_normalize(x, axis=1)

    @staticmethod
    def _get_sum_of_pooling_convolutions_head(x, batch_of_channels_norms):

        # Compute sums across with and height for each channels.
        # Output is a 2D matrix, each row represents channels sums for a single sample
        batch_of_sums_for_each_channel = tf.reduce_sum(x, axis=(1, 2))

        # Normalize by channels norms
        return batch_of_sums_for_each_channel / batch_of_channels_norms

    @staticmethod
    def _get_maximum_activation_of_convolutions_head(x, batch_of_channels_norms):

        batch_of_maximum_activations_for_each_channel = tf.reduce_max(x, axis=(1, 2))

        # Normalize by channels norms
        return batch_of_maximum_activations_for_each_channel / batch_of_channels_norms

    @staticmethod
    def _get_generalized_mean_pooling_head(x, batch_of_channels_norms):

        # Compute mean pooling by first raising elements to power 3, summing up within each channel,
        # normalizing, then taking root of power 3
        scaled_up_elements = tf.math.pow(x, 3)
        summed_up_elements = tf.reduce_sum(scaled_up_elements, axis=(1, 2))
        normalized_elements = summed_up_elements / batch_of_channels_norms

        return tf.math.pow(normalized_elements, 1.0 / 3.0)


class HardAwarePointToSetLossBuilder:
    """
    A helper class for building hard aware point to set loss ops
    """

    @staticmethod
    def get_points_to_sets_losses_op(distances_matrix_op, mask_op, exponential_scaling_constant):
        """
        Get points to sets losses vector op for points/sets specified by mask_op.

        :param distances_matrix_op: 2D tensor op with distances from queries to images in batch,
        each row represents distances from one query to all images in a batch
        :param mask_op: 2D tensor with 1s for elements that should be used in computations and 0s for elements
        that should be masked
        :param exponential_scaling_constant: float, value by which distances are scaled for weights computations
        :return: 1D tensor of weighted point to scale distances, each element represents weighted sum of distances
        between a query and all the non-masked elements from image set
        """

        # Compute weights, after computing multiply by mask, so that any elements that shouldn't be included
        # in computations have their weights zeroed out
        weights_op = tf.math.exp(distances_matrix_op / exponential_scaling_constant) * mask_op

        weighted_distances_op = distances_matrix_op * weights_op

        normalized_weighted_points_to_sets_distances = \
            tf.math.reduce_sum(weighted_distances_op, axis=1) / (tf.math.reduce_sum(weights_op, axis=1) + 1e-6)

        return normalized_weighted_points_to_sets_distances


@tf.function
def get_hard_aware_point_to_set_loss_op(labels, embeddings):
    """
    Implementation of loss from
    "Hard-aware point-to-set deep metric for person re-identification" paper

    :param labels: 1D tensor, batch of labels for embeddings
    :param embeddings: 2D tensor, batch of embeddings
    :return: loss tensor
    """

    if has_any_nan_elements(embeddings):
        tf.print("\nNaN embeddings detected!")

    # Keras adds an unnecessary batch dimension on our labels, flatten them
    flat_labels = tf.reshape(labels, shape=(-1,))

    distances_matrix_op = get_distances_matrix_op(embeddings)

    same_labels_mask = get_vector_elements_equalities_matrix_op(flat_labels)
    diagonal_matrix_op = tf.eye(num_rows=tf.shape(flat_labels)[0], dtype=tf.float32)

    hard_positives_vector_op = HardAwarePointToSetLossBuilder.get_points_to_sets_losses_op(
        distances_matrix_op=distances_matrix_op,
        # Make sure diagonal elements of positives mask are set to zero,
        # so we don't try to set loss on a distance between a vector and itself
        mask_op=same_labels_mask - diagonal_matrix_op,
        exponential_scaling_constant=0.5
    )

    hard_negatives_vector_op = HardAwarePointToSetLossBuilder.get_points_to_sets_losses_op(
        distances_matrix_op=distances_matrix_op,
        # Use negative pairs only
        mask_op=1.0 - same_labels_mask,
        exponential_scaling_constant=-0.5
    )

    # Use soft margin loss instead of hinge loss, as per "In defence of the triplet loss" paper
    losses_vector_op = tf.math.log1p(tf.math.exp(hard_positives_vector_op - hard_negatives_vector_op))

    loss_op = tf.reduce_mean(losses_vector_op)

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

    if has_any_nan_elements(embeddings):
        tf.print("\nNaN embeddings detected!")

    # Keras adds an unnecessary batch dimension on our labels, flatten them
    flat_labels = tf.reshape(labels, shape=(-1,))

    distances_matrix_op = get_distances_matrix_op(embeddings)

    same_labels_mask = get_vector_elements_equalities_matrix_op(flat_labels)
    diagonal_matrix_op = tf.eye(num_rows=tf.shape(flat_labels)[0], dtype=tf.float32)

    positives_mask = same_labels_mask - diagonal_matrix_op

    # For each anchor, select largest distance to same category element
    hard_positives_vector_op = tf.reduce_max(distances_matrix_op * positives_mask, axis=1)

    max_distance_op = tf.reduce_max(distances_matrix_op)

    # Modify distances matrix so that all distances between same labels are set higher than all
    # distances between different labels
    distances_matrix_op_with_distances_between_same_labels_maxed_out = \
        distances_matrix_op + (same_labels_mask * max_distance_op)

    hard_negatives_vector_op = tf.reduce_min(distances_matrix_op_with_distances_between_same_labels_maxed_out, axis=1)

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


def has_any_nan_elements(x):
    """
    Check if tensor contains any NaN values

    :param x: tensor
    :rtype: boolean tensor
    """

    return tf.math.reduce_max(tf.cast(tf.math.is_nan(x), tf.float32)) > 0
