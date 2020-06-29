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
