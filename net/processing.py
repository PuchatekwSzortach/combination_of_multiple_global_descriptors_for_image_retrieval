"""
Module with processing code
"""

import numpy as np


def get_image_padded_to_square_size(image):
    """
    Pad image to square size with zeros

    :param image: image
    :type image: 3D numpy array
    :return: 3D numpy array
    """

    height, width = image.shape[:2]

    height_padding = width - height if width > height else 0
    width_padding = height - width if height > width else 0

    padding = [(0, height_padding), (0, width_padding), (0, 0)]

    return np.pad(
        array=image,
        pad_width=padding,
        mode="constant",
        constant_values=0
    )


class ImageProcessor:
    """
    Simple class wrapping up normalization and denormalization routines
    """

    @staticmethod
    def get_normalized_image(image):
        """
        Get normalized image
        :param image: numpy array
        :return: numpy array
        """

        return np.float32(image / 255.0) - 0.5

    @staticmethod
    def get_denormalized_image(image):
        """
        Transform normalized image back to original scale
        :param image: numpy array
        :return: numpy array
        """

        return np.uint8(255 * (image + 0.5))
