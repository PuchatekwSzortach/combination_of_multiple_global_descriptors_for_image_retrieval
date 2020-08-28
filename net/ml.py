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
    x = tf.keras.layers.BatchNormalization(name="logits")(x)
    x = tf.keras.layers.Dense(units=categories_count, activation=tf.nn.softmax, name="softmax_predictions")(x)

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

        embeddings_head = tf.keras.layers.Dense(units=512, activation=None)(x)

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
                auxiliary_categorization_head_name:
                    get_auxiliary_head_categorization_loss(temperature=0.5)
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
                auxiliary_categorization_head_name:
                    get_auxiliary_head_categorization_loss(temperature=0.5)
            },
            metrics={
                embeddings_head_name: average_ranking_position,
                auxiliary_categorization_head_name: "accuracy"
            }
        )

        return model

    @staticmethod
    def _get_normalized_branch(x, target_size):

        x = tf.keras.layers.Dense(units=target_size, activation=None)(x)
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

        # Keep exponent from being too large, otherwise weights explode to infinity when
        # we compute exponentials
        weights_exponent = tf.minimum(distances_matrix_op / exponential_scaling_constant, 50.0)

        # Compute weights, after computing multiply by mask, so that any elements that shouldn't be included
        # in computations have their weights zeroed out
        weights_op = tf.math.exp(weights_exponent) * mask_op

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

    # Compute norm of each vector. Since derivative of norm of 0 is inifinity, set minimum value to epsilon
    # Credit for noticing need to do so goes to
    # Olivier Moindro, who described the problem here: https://omoindrot.github.io/triplet-loss
    distances = tf.maximum(tf.norm(differences, axis=1), epsilon)

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

    # Normalize vectors, set smallest value after normalization to epsilon, so we avoid infinite gradients
    # on normalization tensor
    y = tf.maximum(tf.math.l2_normalize(x, axis=1), epsilon)
    return y


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

    return tf.math.reduce_any(tf.math.is_nan(x))


def has_any_inf_elements(x):
    """
    Check if tensor contains any inf values

    :param x: tensor
    :rtype: boolean tensor
    """

    return tf.math.reduce_any(tf.math.is_inf(x))


def has_near_zero_element(x):
    """
    Check if tensor contains any near zero values

    :param x: tensor
    :rtype: boolean tensor
    """

    epsilon = 1e-6

    is_smaller_than_epsilon = tf.abs(x) < epsilon

    return tf.math.reduce_any(is_smaller_than_epsilon)


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


def get_temperature_scaled_softmax_cross_entropy_loss_function(temperature):
    """
    Function that builds a softmax cross entropy with specified temperature scaling

    :param temperature: float, temperature to use for temperature scaling
    :return: loss function that accepts two parameters, labels and predictions, and returns scalar loss
    """

    def get_loss(labels, predictions_op):

        # # We want logits, not predictions, so first get them
        logits = predictions_op.op.inputs[0]
        scaled_logits = logits / temperature

        return tf.keras.losses.sparse_categorical_crossentropy(labels, scaled_logits, from_logits=True, axis=-1)

    return get_loss


def get_auxiliary_head_categorization_loss(temperature):
    """
    Function that builds a softmax cross entropy loss with temperature scaling and label smoothing

    :param temperature: float, temperature to use for temperature scaling
    :return: loss function that accepts two parameters, labels and predictions, and returns scalar loss
    """

    def get_loss(labels, predictions_op):

        # First compute temperature scaled logits
        logits = predictions_op.op.inputs[0]
        scaled_logits = logits / temperature

        # Then compute smoothed labels
        one_hot_encoded_labels = tf.one_hot(tf.cast(tf.squeeze(labels), tf.int32), predictions_op.shape[-1])

        return tf.keras.losses.categorical_crossentropy(
            y_true=one_hot_encoded_labels,
            y_pred=scaled_logits,
            from_logits=True,
            label_smoothing=0.1)

    return get_loss
