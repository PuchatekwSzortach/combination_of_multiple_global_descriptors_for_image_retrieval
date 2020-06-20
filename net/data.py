"""
Module with data related code
"""

import scipy.io

import net.constants


class Cars196Annotation:
    """
    Class for representing one sample of Cars196 dataset
    """

    def __init__(self, annotation_matrix, categories_names):
        """
        Constructor

        :param annotation_matrix: annotation for a single sample from Cars196 dataset's loaded from official annotations
        matlab mat file
        :type annotation_matrix: numpy array
        :param categories_names: list of arrays, each array contains a single element,
        string representing category label
        """

        self.filename = str(annotation_matrix[0][0])
        self.category_id = annotation_matrix[-2][0][0] - 1
        self.category = categories_names[self.category_id][0]

        self.dataset_mode = net.constants.DatasetMode(annotation_matrix[-1])


class Cars196DataLoader:
    """
    Data loader class for cars 196 dataset
    """

    def __init__(self, data_dir, annotations_path, dataset_mode):
        """
        Constructor

        :param data_dir: path to base data directory
        :type images_dir: str
        :param annotations_path: path to annotation sdata
        :type annotations_path: str
        :param dataset_mode: net.constants.DatasetMode instance,
        indicates which dataset (train/validation) loader should load
        """

        annotations_data_map = scipy.io.loadmat(annotations_path)

        annotations_matrices = annotations_data_map["annotations"].flatten()
        categories_names = annotations_data_map["class_names"].flatten()

        # Get a list of annotations for specified dataset mode
        self.annotations = [
            Cars196Annotation(
                annotation_matrix=annotation_matrix,
                categories_names=categories_names) for annotation_matrix in annotations_matrices
            if net.constants.DatasetMode(annotation_matrix[-1][0][0]) == dataset_mode]

        self.data_dir = data_dir
        self.dataset_model = dataset_mode
