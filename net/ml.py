"""
Module with machine learning code
"""

import tensorflow as tf


def get_auxiliary_categorization_head(x, categories_count):
    """
    Get a simple categorization head

    :param x: 2D tensor op, batch of 1D vectors
    :param categories_count: int, number of categories for auxiliary categorization head's output
    :return: 2D tensor op, final layer uses softmax activation
    """

    x = tf.keras.layers.Dense(units=categories_count, activation=tf.nn.swish)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(units=categories_count, activation=tf.nn.softmax)(x)

    return x


class ImagesSimilarityComputer:
    """
    Class for computing similarity between images.
    """

    @staticmethod
    def get_model(image_size, categories_count):
        """
        Model builder

        :param image_size: int, height and width of input image for the model
        :param categories_count: int, number of categories for auxiliary categorization head's output
        :return: keras.Model instance
        """

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3)
        )

        input_op = base_model.input

        x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), activation=tf.nn.swish)(base_model.output)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), activation=tf.nn.swish)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1024, activation=tf.nn.swish)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        common_backbone = x

        embeddings_head = tf.keras.layers.Dense(units=128, activation=None)(common_backbone)

        auxiliary_categorization_head = \
            get_auxiliary_categorization_head(x=embeddings_head, categories_count=categories_count)

        embeddings_head_name = "embeddings"
        auxiliary_categorization_head_name = "auxiliary_categorization_head"

        embeddings_head = tf.keras.layers.Lambda(
            lambda x: x,
            name=embeddings_head_name)(embeddings_head)

        auxiliary_categorization_head = tf.keras.layers.Lambda(
            lambda x: x,
            name=auxiliary_categorization_head_name)(auxiliary_categorization_head)

        model = tf.keras.models.Model(
            inputs=input_op,
            outputs=[embeddings_head, auxiliary_categorization_head]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                embeddings_head_name: get_hard_aware_point_to_set_loss_op,
                auxiliary_categorization_head_name: "sparse_categorical_crossentropy"
            },
            metrics={
                embeddings_head_name: average_ranking_position,
                auxiliary_categorization_head_name: "accuracy"
            }
        )

        return model


class CGDImagesSimilarityComputer:
    """
    Class for computing similarity between images based on Combination of Multiple Global Descriptors model
    """

    @staticmethod
    def get_model(image_size, categories_count):
        """
        Model builder

        :param image_size: int, height and width of input image for the model
        :param categories_count: int, number of categories for auxiliary categorization head's output
        :return: keras.Model instance
        """

        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(image_size, image_size, 3)
        )

        input_op = base_model.input

        x = base_model.output

        sum_of_pooling_convolutions_features = CGDImagesSimilarityComputer._get_normalized_branch(
            x=CGDImagesSimilarityComputer._get_sum_of_pooling_convolutions_head(x),
            target_size=512)

        maximum_activations_of_convolutions_features = CGDImagesSimilarityComputer._get_normalized_branch(
            x=CGDImagesSimilarityComputer._get_maximum_activation_of_convolutions_head(x),
            target_size=512)

        generalized_mean_pooling_features = CGDImagesSimilarityComputer._get_normalized_branch(
            x=CGDImagesSimilarityComputer._get_generalized_mean_pooling_head(x),
            target_size=512)

        combination_of_multiple_global_descriptors = tf.concat(
            [
                sum_of_pooling_convolutions_features,
                maximum_activations_of_convolutions_features,
                generalized_mean_pooling_features
            ],
            axis=1)

        embeddings_head_name = "embeddings"

        embeddings_head = tf.keras.layers.Lambda(
            lambda x: x,
            name=embeddings_head_name)(l2_normalize_batch_of_vectors(combination_of_multiple_global_descriptors))

        auxiliary_categorization_head_name = "auxiliary_categorization_head"

        auxiliary_categorization_head = \
            get_auxiliary_categorization_head(
                x=sum_of_pooling_convolutions_features,
                categories_count=categories_count)

        auxiliary_categorization_head = tf.keras.layers.Lambda(
            lambda x: x,
            name=auxiliary_categorization_head_name)(auxiliary_categorization_head)

        model = tf.keras.models.Model(
            inputs=input_op,
            outputs=[embeddings_head, auxiliary_categorization_head]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                embeddings_head_name: get_hard_aware_point_to_set_loss_op,
                auxiliary_categorization_head_name: "sparse_categorical_crossentropy"
            },
            metrics={
                embeddings_head_name: average_ranking_position,
                auxiliary_categorization_head_name: "accuracy"
            }
        )

        return model

    @staticmethod
    def _get_normalized_branch(x, target_size):

        x = tf.keras.layers.Dense(units=target_size, activation=tf.nn.swish)(x)
        return l2_normalize_batch_of_vectors(x)

    @staticmethod
    def _get_sum_of_pooling_convolutions_head(x):

        return tf.reduce_mean(x, axis=(1, 2))

    @staticmethod
    def _get_maximum_activation_of_convolutions_head(x):

        return tf.reduce_max(x, axis=(1, 2))

    @staticmethod
    def _get_generalized_mean_pooling_head(x):

        # Compute mean pooling by first raising elements to power 3, computing mean, then taking cubic root of result
        scaled_up_elements = tf.math.pow(x, 3)
        channels_means = tf.reduce_mean(scaled_up_elements, axis=(1, 2))
        return tf.math.pow(channels_means, 1.0 / 3.0)


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
    Given a 2D matrix tensor, return euclidean distance between each row vector
    This implementation tries to take care of cases when two rows are identical in a way that yields stable (zeo)
    gradients.

    :param matrix_op: 2D tensor of row vectors
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

    epsilon = tf.constant(1e-6, tf.float32)

    # Compute mask set to 1 for each row that has vector with non-zero length and 0 otherwise
    nonzero_vectors_mask = tf.cast(tf.math.abs(tf.reduce_sum(differences, axis=1)) > epsilon, tf.float32)

    # differences masked so that vectors that would have zero norm/are made of 0 elements have epsilon added to them.
    differences_with_zero_vectors_set_to_epsilon = \
        differences + (epsilon * (1.0 - tf.reshape(nonzero_vectors_mask, (-1, 1))))

    # Compute norm of each vector.
    # This computation requires taking square root of sum of squares of vector elements.
    # And computing gradients for it requires a division by square root.
    # If distance for any vector is 0, which can only happen if all its elements are 0,
    # then we would be dividing by square root of 0, which is 0.
    # To avoid this we add a small epsilon to elements that would have 0 norms before norm computations.
    # We will then reset these values back to 0 afterwards.
    # Credit for noticing need to do so goes to
    # Olivier Moindro, who described the problem here: https://omoindrot.github.io/triplet-loss
    distances_with_zero_values_set_to_epsilon = tf.norm(differences_with_zero_vectors_set_to_epsilon, axis=1)

    # Now remove epsilon elements from distance vector
    distances = distances_with_zero_values_set_to_epsilon * nonzero_vectors_mask

    # So reshape it to a matrix of same shape as input
    return tf.reshape(
        tensor=distances,
        shape=(rows_count_op, rows_count_op)
    )


def l2_normalize_batch_of_vectors(x):
    """
    Given a matrix representing group of vectors, one vector per row, l2 normalize each vector.
    This implementation wraps tf.math.l2_normalize with ops that try to make sure that should any vector
    be 0, then its gradient will also be 0 instead of infinite.

    :param x: 2D tensorflow op
    :return: 2D tensorflow op
    """

    epsilon = 1e-6

    # Get a mask indicating which vectors are non-zero and which aren't
    nonzero_vectors_mask = tf.cast(tf.reduce_sum(tf.abs(x), axis=1) > epsilon, tf.float32)

    # Reshape mask so it gets broadcasted across vectors during computations with them
    reshaped_nonzero_vectors_mask = tf.reshape(nonzero_vectors_mask, (-1, 1))

    # Modify x so that vectors that are zero have their elements set to epsilon instead
    x_modified = x + (epsilon * (1.0 - reshaped_nonzero_vectors_mask))

    y = tf.math.l2_normalize(x_modified, axis=1)

    # Now for vectors that had epsilons added to them set them back to 0.
    # The end result is that norms are as with tf.math.l2_normalize, but gradients at vectors that were 0
    # are 0, instead of infinite/very high number
    y_modified = y * reshaped_nonzero_vectors_mask

    return y_modified


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


def average_ranking_position(labels, embeddings):
    """
    Compute average ranking position of correct label images for each query image.

    :param labels: [n x 1] tensor with labels
    :param embeddings: 2D tensor with embeddings, each row represents embeddings for a single input
    """

    # Keras adds an unnecessary batch dimension on our labels, flatten them
    flat_labels = tf.reshape(labels, shape=(-1,))

    distances_matrix_op = get_distances_matrix_op(embeddings)

    same_labels_mask = get_vector_elements_equalities_matrix_op(flat_labels)

    # For each element in a row get index it would have in sorted array, or its distance rank.
    # To compute this value we run argsort twice.
    # First run returns indices of elements in sorted order.
    # Second argsort returns index into this array for each original element,
    # in effect giving us rank values for each element.
    distances_rankings = tf.argsort(tf.argsort(distances_matrix_op, axis=1), axis=1)

    # Set distance ranking for samples that don't have same label as query label to zero
    distances_rankings_with_negative_samples_distances_set_to_zero = \
        (same_labels_mask) * tf.cast(distances_rankings, tf.float32)

    # To compute average rank for each query:
    # - compute sum of ranks for all samples with that query
    # - divide by number of elements with same labels as query
    # Since any negative queries had their distance ranks set to 0, they will not contribute to the sum,
    # therefore yielding correct results
    per_label_average_rank = \
        tf.reduce_sum(distances_rankings_with_negative_samples_distances_set_to_zero, axis=1) / \
        tf.reduce_sum(same_labels_mask, axis=1)

    return tf.reduce_mean(per_label_average_rank)
