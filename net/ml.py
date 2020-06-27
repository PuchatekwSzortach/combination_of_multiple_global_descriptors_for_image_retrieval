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
            weights="imagenet"
        )

        self.input = base_model.input
        self.output = base_model.output

        self.model = tf.keras.models.Model(
            inputs=self.input,
            outputs=self.output
        )
