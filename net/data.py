"""
Module with data related code
"""

import collections
import os
import random

import cv2
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


def get_cars_196_annotations_map(annotations_path, dataset_mode):
    """
    Read cars 196 annotations into a category_id: list of Cars196Annotation map and return it

    :param annotations_path: path to annotations data
    :type annotations_path: str
    :param dataset_mode: net.constants.DatasetMode instance,
    indicates annotations for which dataset (train/validation) should be loaded
    :return: map {category_id: list of Cars196Annotation}
    :rtype: dict
    """

    annotations_data_map = scipy.io.loadmat(annotations_path)

    annotations_matrices = annotations_data_map["annotations"].flatten()
    categories_names = annotations_data_map["class_names"].flatten()

    # Get a list of annotations for specified dataset mode
    annotations = [
        Cars196Annotation(
            annotation_matrix=annotation_matrix,
            categories_names=categories_names) for annotation_matrix in annotations_matrices
        if net.constants.DatasetMode(annotation_matrix[-1][0][0]) == dataset_mode]

    categories_ids_samples_map = collections.defaultdict(list)

    # Move annotations into categories_ids: annotations map
    for annotation in annotations:

        categories_ids_samples_map[annotation.category_id].append(annotation)

    return categories_ids_samples_map


class Cars196DataLoader:
    """
    Data loader class for cars 196 dataset
    """

    def __init__(self, data_dir, annotations_path, dataset_mode, categories_per_batch, samples_per_category):
        """
        Constructor

        :param data_dir: path to base data directory
        :type images_dir: str
        :param annotations_path: path to annotations data
        :type annotations_path: str
        :param dataset_mode: net.constants.DatasetMode instance,
        indicates which dataset (train/validation) loader should load
        :param categories_per_batch: number of categories in each batch
        :type categories_per_batch: int
        :param samples_per_category: number of samples for a category in a batch
        :type samples_per_category: int
        """

        self.categories_ids_samples_map = get_cars_196_annotations_map(
            annotations_path=annotations_path,
            dataset_mode=dataset_mode)

        self.data_dir = data_dir

        self.categories_per_batch = categories_per_batch
        self.samples_per_category = samples_per_category

    def __iter__(self):

        while True:

            categories_samples_batch = self._get_categories_samples_batch()

            categories_images_batch = collections.defaultdict(list)
            categories_labels_batch = collections.defaultdict(list)

            for category, samples in categories_samples_batch.items():

                for sample in samples:

                    image = cv2.imread(os.path.join(self.data_dir, sample.filename))

                    categories_images_batch[category].append(image)
                    categories_labels_batch[category].append(sample.category_id)

            yield categories_images_batch, categories_labels_batch

    def _get_categories_samples_batch(self):
        """
        Draw a batch of categories samples - k categories, p samples for each
        :return: dictionary {category_id: list of Cars196Annotation instances}
        """

        # Select categories to draw from
        categories_to_draw = random.sample(
            population=self.categories_ids_samples_map.keys(),
            k=self.categories_per_batch)

        batch_map = {}

        for category in categories_to_draw:

            samples_to_draw = random.sample(
                population=self.categories_ids_samples_map[category],
                k=self.samples_per_category
            )

            batch_map[category] = samples_to_draw

        return batch_map
