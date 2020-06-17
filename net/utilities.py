"""
Module with utilities
"""

import logging
import os

import yaml


def read_yaml(path):
    """Read content of yaml file from path

    :param path: path to yaml file
    :type path: str
    :return: yaml file content, usually a dictionary
    """

    with open(path) as file:

        return yaml.safe_load(file)


def get_logger(path):
    """
    Returns a logger configured to write to a file
    :param path: path to file logger should write to
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("image_retrieval")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
