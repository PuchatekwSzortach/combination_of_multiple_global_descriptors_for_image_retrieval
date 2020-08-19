"""
Module with processing code
"""

import cv2
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

    top_padding = height_padding // 2
    bottom_padding = top_padding if height_padding % 2 == 0 else top_padding + 1

    left_padding = width_padding // 2
    right_padding = left_padding if width_padding % 2 == 0 else left_padding + 1

    padding = [(top_padding, bottom_padding), (left_padding, right_padding), (0, 0)]

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
    def get_resized_image(image, target_size):
        """
        Resize image to common format.
        First pads image to square size, then resizes it to target_size x target_size

        :param image: 3D numpy array
        :param target_size: int, size to which image should be resized
        :return: 3D numpy array
        """

        image = get_image_padded_to_square_size(image)
        image = cv2.resize(image, (target_size, target_size))

        return image

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
