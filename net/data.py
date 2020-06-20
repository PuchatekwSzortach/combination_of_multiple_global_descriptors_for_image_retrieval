"""
Module with data related code
"""

import scipy.io

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


class Cars196DataLoader:
    """
    Data loader class for cars 196 dataset
    """

    def __init__(self, images_dir, annotations_path, dataset_mode):
        """
        Constructor

        :param images_dir: path do directory with images
        :type images_dir: str
        :param annotations_path: path to annotation sdata
        :type annotations_path: str
        :param dataset_mode: net.constants.DatasetMode instance,
        indicates which dataset (train/validation) loader should load
        """

        self.annotations_data_map = scipy.io.loadmat(annotations_path)
        self.images_dir = images_dir
        self.dataset_model = dataset_mode
