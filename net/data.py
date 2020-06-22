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

        self.data_dir = data_dir

        self.categories_ids_samples_map = get_cars_196_annotations_map(
            annotations_path=annotations_path,
            dataset_mode=dataset_mode)

        self.dataset_mode = dataset_mode

        self.categories_per_batch = categories_per_batch
        self.samples_per_category = samples_per_category

    def __iter__(self):

        while True:

            samples_batches_drawer = net.data.SamplesBatchesDrawer(
                categories_samples_map=self.categories_ids_samples_map,
                categories_per_batch=self.categories_per_batch,
                samples_per_category=self.samples_per_category,
                shuffle=self.dataset_mode == net.constants.DatasetMode.TRAINING
            )

            for categories_samples_batch in samples_batches_drawer:

                categories_images_batch = collections.defaultdict(list)
                categories_labels_batch = collections.defaultdict(list)

                for category, samples in categories_samples_batch.items():

                    for sample in samples:

                        image = cv2.imread(os.path.join(self.data_dir, sample.filename))

                        categories_images_batch[category].append(image)
                        categories_labels_batch[category].append(sample.category_id)

                yield categories_images_batch, categories_labels_batch


class SamplesBatchesDrawer:
    """
    Class for drawing samples batches from a dictionary with {category: samples} structure
    """

    def __init__(self, categories_samples_map, categories_per_batch, samples_per_category, shuffle):
        """
        Constructor

        :param categories_samples_map: data to draw samples from, map with format {category: samples}
        :type categories_samples_map: dict
        :param categories_per_batch: number of categories to be included in a batch
        :type categories_per_batch: int
        :param samples_per_category: number of samples for each category to be included in a batch
        :type samples_per_category: int
        :param shuffle: bool, specifies if data should be shuffled before drawing
        """

        self.categories_samples_map = categories_samples_map
        self.categories_per_batch = categories_per_batch
        self.samples_per_category = samples_per_category
        self.shuffle = shuffle

    def __iter__(self):

        # Compute a {categories: samples indices} map
        categories_samples_indices_map = {
            category: np.arange(len(samples))
            for category, samples in self.categories_samples_map.items()}

        if self.shuffle is True:

            # Shuffle samples indices - we will the draw from shuffled indices list sequentially to simulate
            # shuffling samples
            for samples_indices in categories_samples_indices_map.values():
                random.shuffle(samples_indices)

        # Set of categories to be used for drawing samples
        categories_pool = set(categories_samples_indices_map.keys())

        for _ in range(len(self)):

            # Pick categories for the batch
            # If shuffling is True, then randomly sample categories from categories pool.
            # Otherwise choose categories in sorted order
            categories_to_draw = random.sample(
                population=categories_pool,
                k=self.categories_per_batch
            ) if self.shuffle is True else sorted(categories_pool)[:self.categories_per_batch]

            batch = {}

            # Pick samples for categories in the batch
            for category in categories_to_draw:

                samples_indices = categories_samples_indices_map[category]

                # Pick a batch of samples indices, remove it from the indices list
                batch_samples_indices = samples_indices[:self.samples_per_category]
                categories_samples_indices_map[category] = samples_indices[self.samples_per_category:]

                # Using samples indices pick samples, store them in batch
                batch[category] = self.categories_samples_map[category][batch_samples_indices]

                # If category has less samples left than we draw per batch, pop that category from
                # categories pool
                if len(categories_samples_indices_map[category]) < self.samples_per_category:
                    categories_pool.remove(category)

            yield batch

    def __len__(self):

        # Compute a lower bound on possible number of batches
        lowest_samples_count = min([len(samples) for samples in self.categories_samples_map.values()])

        # A low bound on number of batches we can draw for a single category
        batches_per_category = lowest_samples_count // self.samples_per_category

        categories_count = len(self.categories_samples_map.keys())

        # Each batch draws samples_per_category samples from categories_per_batch categories.
        # That means that we have (categories_count // categories_per_batch) unique sets
        # of categories from which batches can be draw.
        # Thus total lower bound of batches is:
        return batches_per_category * categories_count // self.categories_per_batch
