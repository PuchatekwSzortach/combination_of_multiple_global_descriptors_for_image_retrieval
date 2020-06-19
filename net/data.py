"""
Module with data related code
"""

import net.constants


class Cars196Annotation:
    """
    Class for representing one sample of Cars196 dataset
    """

    def __init__(self, annotation_matrix):
        """
        Constructor

        :param annotation_matrix: annotation for a single sample from Cars196 dataset's loaded from official annotations
        matlab mat file
        :type annotation_matrix: numpy array
        """

        self.filename = str(annotation_matrix[0][0])
        self.category_id = annotation_matrix[-2][0][0] - 1

        self.dataset_mode = net.constants.DatasetMode(annotation_matrix[-1])
