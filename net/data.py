"""
Module with data related code
"""

import collections
import os
import random

import cv2
import numpy as np
import scipy.io

import net.constants
import net.processing


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
        self.category_id = int(annotation_matrix[-2][0][0] - 1)
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

    # Convert lists of samples to np.arrays of samples
    return {category_id: np.array(samples) for category_id, samples in categories_ids_samples_map.items()}


class Cars196DataLoader:
    """
    Data loader class for cars 196 dataset
    """

    # def __init__(self, data_dir, annotations_path, dataset_mode, categories_per_batch, samples_per_category):
    def __init__(self, config, dataset_mode):
        """
        Constructor

        :param config: dictionary with data loader configuration
        :param dataset_mode: net.constants.DatasetMode instance,
        indicates which dataset (train/validation) loader should load
        """

        self.data_dir = config["data_dir"]

        self.categories_ids_samples_map = get_cars_196_annotations_map(
            annotations_path=config["annotations_path"],
            dataset_mode=dataset_mode)

        self.dataset_mode = dataset_mode

        self.categories_per_batch = config["train"]["categories_per_batch"]
        self.samples_per_category = config["train"]["samples_per_category"]

        self.image_size = config["image_size"]

    def __iter__(self):

        while True:

            samples_batches_drawer = SamplesBatchesDrawer(
                categories_samples_map=self.categories_ids_samples_map,
                categories_per_batch=self.categories_per_batch,
                samples_per_category=self.samples_per_category,
                dataset_mode=self.dataset_mode
            )

            for samples_batch, categories_batch in samples_batches_drawer:

                images_batch = [
                    self.get_processed_image(
                        image=cv2.imread(os.path.join(self.data_dir, sample.filename)),
                        target_size=self.image_size) for sample in samples_batch
                ]

                yield np.array(images_batch), np.array(categories_batch)

    @staticmethod
    def get_processed_image(image, target_size):
        """
        Process image - pad to square, scale, etc

        :param image: 3D numpy array
        :param target_size: int, size to which image should be adjusted
        :return: 3D numpy array
        """

        image = net.processing.get_image_padded_to_square_size(image)
        image = cv2.resize(image, (target_size, target_size))

        return net.processing.ImageProcessor.get_normalized_image(image)

    def __len__(self):

        samples_batches_drawer = net.data.SamplesBatchesDrawer(
            categories_samples_map=self.categories_ids_samples_map,
            categories_per_batch=self.categories_per_batch,
            samples_per_category=self.samples_per_category,
            dataset_mode=self.dataset_mode
        )

        return len(samples_batches_drawer)


class SamplesBatchesDrawer:
    """
    Class for drawing samples batches from a dictionary with {category: samples} structure.
    It yields (samples_batch, categories_batch) tuples.
    Input data might contain different number of samples for different categories,
    but generator will yield for each category only number of batches equal to number of batches of unique samples
    for the category with smallest number of batches.
    """

    def __init__(self, categories_samples_map, categories_per_batch, samples_per_category, dataset_mode):
        """
        Constructor

        :param categories_samples_map: data to draw samples from, map with format {category: samples}
        :type categories_samples_map: dict
        :param categories_per_batch: number of categories to be included in a batch
        :type categories_per_batch: int
        :param samples_per_category: number of samples for each category to be included in a batch
        :type samples_per_category: int
        :param dataset_mode: net.constants.DatasetMode instance.
        If training mode is used both categories and samples are shuffled randomly.
        Otherwise only categories are shuffled and constant random seed is used, so that results are repeatable
        across runs.
        """

        self.categories_samples_map = categories_samples_map
        self.categories_per_batch = categories_per_batch
        self.samples_per_category = samples_per_category
        self.dataset_mode = dataset_mode

        # Random number generator. Use random seed if we are in training mode, otherwise use constant seed
        self.random = random.Random() if dataset_mode is net.constants.DatasetMode.TRAINING else random.Random(0)

        self.lowest_samples_count = min([len(samples) for samples in self.categories_samples_map.values()])
        self.max_batches_count_in_single_category = self.lowest_samples_count // self.samples_per_category

    def __iter__(self):

        categories_samples_indices_map = self._get_categories_indices_map()

        # Set of categories to be used for drawing samples
        categories_pool = set(categories_samples_indices_map.keys())

        for _ in range(len(self)):

            # Pick categories for the batch
            categories_to_draw = self.random.sample(
                population=categories_pool,
                k=self.categories_per_batch
            )

            samples_batch = []
            categories_labels_batch = []

            # Pick samples for categories in the batch
            for category in categories_to_draw:

                samples_indices = categories_samples_indices_map[category]

                # Pick a batch of samples indices, remove it from the samples indices list
                samples_indices_batch = samples_indices[:self.samples_per_category]
                categories_samples_indices_map[category] = samples_indices[self.samples_per_category:]

                # Using samples indices pick samples, store them in batch
                samples_batch.extend(self.categories_samples_map[category][samples_indices_batch])
                categories_labels_batch.extend([category] * self.samples_per_category)

                # If we already drew max number of batches from this category, remove it from categories pool
                if len(categories_samples_indices_map[category]) < self.samples_per_category:

                    categories_pool.remove(category)
                    categories_samples_indices_map.pop(category)

            yield samples_batch, categories_labels_batch

    def __len__(self):

        # Compute into how many smaller, independent subsets can we divide data
        categories_count = len(self.categories_samples_map.keys())
        independent_subdatasets_count = categories_count // self.categories_per_batch

        # The number of batches we can support is number of batches in a single smaller sub-dataset
        # times number of such smaller sub-datasets minus a small number for categories that can be
        # depleted sooner due to possible straddled draws
        return (independent_subdatasets_count * self.max_batches_count_in_single_category) - 1

    def _get_categories_indices_map(self):
        """
        Based on {category: samples} dataset instance is initialized with,
        compute a {category: samples indices} dictionary.
        For every category only number of samples equal to smallest number of samples across all categories is used.
        If instance was initialized in training mode, samples indices are shuffled before truncathing them to
        common size.
        """

        categories_samples_indices_map = {
            category: np.arange(len(samples))
            for category, samples in self.categories_samples_map.items()}

        if self.dataset_mode is net.constants.DatasetMode.TRAINING:

            # Shuffle samples indices - we will then draw from shuffled indices list sequentially to simulate
            # shuffling samples
            for samples_indices in categories_samples_indices_map.values():
                self.random.shuffle(samples_indices)

        # Since we want to make distribution between categories uniform, truncate number of samples indices
        # in each category to number of samples in category with least samples
        truncated_categories_samples_indices_map = {
            category: samples_indices[:self.lowest_samples_count]
            for category, samples_indices in categories_samples_indices_map.items()
        }

        return truncated_categories_samples_indices_map
