"""
Module with constants
"""

import enum


class DatasetMode(enum.Enum):
    """
    Simple enum to differentiate between training and validation datasets
    """

    TRAINING = 0
    VALIDATION = 1
