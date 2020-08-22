"""
Module with data related code
"""

import collections
import functools
import os
import random

import cv2
import imgaug
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

        self.dataset_mode = \
            net.constants.DatasetMode.TRAINING if self.category_id <= 98 else net.constants.DatasetMode.VALIDATION


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

    # Get a list of annotations
    annotations = [
        Cars196Annotation(
            annotation_matrix=annotation_matrix,
            categories_names=categories_names) for annotation_matrix in annotations_matrices]

    # Filter annotations by target dataset mode
    filtered_annotations = [annotation for annotation in annotations if annotation.dataset_mode == dataset_mode]

    categories_ids_samples_map = collections.defaultdict(list)

    # Move annotations into categories_ids: annotations map
    for annotation in filtered_annotations:

        categories_ids_samples_map[annotation.category_id].append(annotation)

    # Convert lists of samples to np.arrays of samples
    return {category_id: np.array(samples) for category_id, samples in categories_ids_samples_map.items()}


class Cars196AnalysisDataLoader:
    """
    Data loader for cars 196 dataset.
    This data loader loads and yields batches of (images, labels) in order they are read from disk.
    It returns each sample exactly once, but makes no attempt to shuffle or balance categories returned in each batch.
    """

    def __init__(self, config, dataset_mode):
        """
        Constructor

        :param config: dictionary with data loader configuration
        :param dataset_mode: net.constants.DatasetMode instance,
        indicates which dataset (train/validation) loader should load
        """

        self.data_dir = config["data_dir"]
        self.image_size = config["image_size"]

        categories_ids_samples_map = get_cars_196_annotations_map(
            annotations_path=config["annotations_path"],
            dataset_mode=dataset_mode)

        # Extract a flat list of samples from categories_ids_samples_map
        self.annotations = []

        for annotations_for_single_category in categories_ids_samples_map.values():

            self.annotations.extend(annotations_for_single_category)

        self.batch_size = 32

    def __iter__(self):

        iterator = self.get_verbose_iterator()

        while True:

            _, images_batch, labels_batch = next(iterator)
            yield images_batch, labels_batch

    def get_verbose_iterator(self):
        """
        Get iterator that yields (images_paths, images, labels) batches
        """

        index = 0

        while index < len(self.annotations):

            yield self._get_verbose_batch(index)
            index += self.batch_size

    def _get_verbose_batch(self, start_index):

        annotations_batch = self.annotations[start_index: start_index + self.batch_size]

        images_paths_batch = [os.path.join(self.data_dir, annotation.filename) for annotation in annotations_batch]

        images_batch = [
            net.processing.ImageProcessor.get_resized_image(
                image=cv2.imread(image_path),
                target_size=self.image_size)
            for image_path in images_paths_batch
        ]

        images_batch = [net.processing.ImageProcessor.get_normalized_image(image) for image in images_batch]
        categories_batch = [annotation.category_id for annotation in annotations_batch]

        return images_paths_batch, np.array(images_batch), np.array(categories_batch)

    def __len__(self):

        return len(self.annotations) // self.batch_size


class Cars196TrainingLoopDataLoader:
    """
    Data loader class for cars 196 dataset
    """

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

        self.image_size = config["image_size"]

        self.augmentations_pipeline = imgaug.augmenters.Sequential(
            children=[
                imgaug.augmenters.SomeOf(
                    n=(0, 2),
                    children=[
                        imgaug.augmenters.Grayscale(alpha=(0.2, 1)),
                        imgaug.augmenters.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                        imgaug.augmenters.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
                        imgaug.augmenters.Affine(rotate=(-15, 15))
                    ],
                    random_order=True),
                # Left-right flip
                imgaug.augmenters.Fliplr(0.5)]
        ) if dataset_mode is net.constants.DatasetMode.TRAINING else None

        self.samples_batches_drawer_builder = functools.partial(
            SamplesBatchesDrawer,
            categories_samples_map=self.categories_ids_samples_map,
            categories_per_batch=config["train"]["categories_per_batch"],
            samples_per_category=config["train"]["samples_per_category"],
            dataset_mode=self.dataset_mode,
            samples_per_category_per_epoch_percentile=100
        )

    def __iter__(self):

        while True:

            samples_batches_drawer = self.samples_batches_drawer_builder()

            for samples_batch, categories_batch in samples_batches_drawer:

                images_batch = [
                    net.processing.ImageProcessor.get_resized_image(
                        image=cv2.imread(os.path.join(self.data_dir, sample.filename)),
                        target_size=self.image_size)
                    for sample in samples_batch
                ]

                if self.dataset_mode is net.constants.DatasetMode.TRAINING:

                    images_batch = self.augmentations_pipeline(images=images_batch)

                images_batch = [net.processing.ImageProcessor.get_normalized_image(image) for image in images_batch]
                yield np.array(images_batch), np.array(categories_batch)

    def __len__(self):

        samples_batches_drawer = self.samples_batches_drawer_builder()
        return len(samples_batches_drawer)


class SamplesBatchesDrawer:
    """
    Class for drawing samples batches from a dictionary with {category: samples} structure.
    It yields (samples_batch, categories_batch) tuples.
    Input data might contain different number of samples for different categories,
    but generator will try to yield from all categories as evenly as possible given input parameters.
    In case of imbalanced number of samples between categories some samples may be yielded more than once
    per epoch.
    """

    def __init__(
            self, categories_samples_map, categories_per_batch, samples_per_category,
            dataset_mode, samples_per_category_per_epoch_percentile):
        """
        Constructor

        :param categories_samples_map: dict, data to draw samples from, map with format {category: samples}
        :param categories_per_batch: int, number of categories to be included in a batch
        :param samples_per_category: int, number of samples for each category to be included in a batch
        :param dataset_mode: net.constants.DatasetMode instance. If training mode is used,
        both categories and samples are shuffled randomly. Otherwise only categories are shuffled and
        constant random seed is used, so that results are repeatable across runs.
        :param samples_per_category_per_epoch_percentile: int, used to decide number of samples to be yielded
        for each category per epoch. Samples count from all categories in categories_samples_map will be computed,
        and percentile from this data will be used to compute samples per epoch for each category in yielded data.
        """

        self.categories_samples_map = categories_samples_map
        self.categories_per_batch = categories_per_batch
        self.samples_per_category = samples_per_category
        self.dataset_mode = dataset_mode

        # Random number generator. Use random seed if we are in training mode, otherwise use constant seed
        self.random = random.Random() if dataset_mode is net.constants.DatasetMode.TRAINING else random.Random(0)

        self.samples_per_category_per_epoch = int(np.percentile(
            [len(samples) for samples in self.categories_samples_map.values()],
            samples_per_category_per_epoch_percentile))

    def __len__(self):

        # Compute into how many smaller, independent subsets can we divide data
        categories_count = len(self.categories_samples_map.keys())
        independent_subdatasets_count = categories_count // self.categories_per_batch

        draws_per_category = self.samples_per_category_per_epoch // self.samples_per_category

        return independent_subdatasets_count * draws_per_category

    def _get_categories_indices_map(self):
        """
        Based on {category: samples} dataset instance is initialized with,
        compute a {category: samples indices} dictionary.
        If instance was initialized in training mode, samples indices are shuffled
        """

        categories_samples_indices_map = {
            category: np.arange(len(samples))
            for category, samples in self.categories_samples_map.items()}

        if self.dataset_mode is net.constants.DatasetMode.TRAINING:

            # Shuffle samples indices - we will then draw from shuffled indices list sequentially to simulate
            # shuffling samples
            for samples_indices in categories_samples_indices_map.values():
                self.random.shuffle(samples_indices)

        return categories_samples_indices_map

    def __iter__(self):

        categories_samples_indices_map = self._get_categories_indices_map()

        # Set of categories to be used for drawing samples
        primary_categories_pool = set(categories_samples_indices_map.keys())

        # If there aren't enough categories in primary pool to satisfy batch requirements,
        # reused categories pool will be used
        reused_categories_pool = set()

        for _ in range(len(self)):

            # Pick categories for the batch
            # If there aren't enough categories available in primary categories pool, draw only as many as we can
            categories_to_draw = self.random.sample(
                population=primary_categories_pool,
                k=min(self.categories_per_batch, len(primary_categories_pool))
            )

            # If primary categories pool couldn't supply enough categories,
            # draw remained from reused categories pool
            if len(categories_to_draw) < self.categories_per_batch:

                categories_to_draw.extend(
                    self.random.sample(
                        population=reused_categories_pool,
                        k=self.categories_per_batch - len(categories_to_draw)
                    )
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

                # If we already drew max number of batches from this category
                if len(categories_samples_indices_map[category]) < self.samples_per_category:

                    # If this category is currently in primary categories pool, removed it from it and add it
                    # to reused categories pool instead
                    if category in primary_categories_pool:

                        primary_categories_pool.remove(category)
                        reused_categories_pool.add(category)

                    # Replenish samples indices for this category
                    samples_indices = np.arange(len(self.categories_samples_map[category]))

                    if self.dataset_mode is net.constants.DatasetMode.TRAINING:

                        self.random.shuffle(samples_indices)

                    categories_samples_indices_map[category] = samples_indices

            yield samples_batch, categories_labels_batch
