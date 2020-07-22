"""
Tests for net.data module
"""

import random

import numpy as np

import net.constants
import net.data


class TestSamplesBatchesDrawer:
    """
    Tests for net.data.SamplesBatchesDrawer class
    """

    def test_len_when_all_categories_have_equal_number_of_samples(self):
        """
        Test len method with data such that all categories have equal number of samples
        """

        samples = np.arange(20)

        categories_samples_map = {
            1: samples,
            2: samples,
            3: samples,
            4: samples
        }

        samples_batches_drawer = net.data.SamplesBatchesDrawer(
            categories_samples_map=categories_samples_map,
            categories_per_batch=2,
            samples_per_category=5,
            dataset_mode=net.constants.DatasetMode.VALIDATION
        )

        # For each category we should be able to draw 4 samples, and we draw 2 categories from total of 4 categories,
        # thus expected to have a total of 8 possible batches full batches. We then remove one for possible
        # effect of straddled draws
        expected = 7
        actual = len(samples_batches_drawer)

        assert expected == actual

        # Also check generator actually yields reported number of batches
        assert expected == len(list(samples_batches_drawer))

    def test_len_when_categories_have_unequal_number_of_samples(self):
        """
        Test len method with data such that some categories have different number of samples than others
        """

        categories_samples_map = {
            1: np.arange(20),
            2: np.arange(15),
            3: np.arange(12),
            4: np.arange(20)
        }

        samples_batches_drawer = net.data.SamplesBatchesDrawer(
            categories_samples_map=categories_samples_map,
            categories_per_batch=2,
            samples_per_category=5,
            dataset_mode=net.constants.DatasetMode.VALIDATION
        )

        # category "3" has 12 samples, so we should be able to take only 2 batches from it.
        # Since we draw 2 categories out of 4 per batch, we then expect to have 2 * 2 = 4 total batches
        # We then remove one for possible effect of straddled drws
        expected = 3
        actual = len(samples_batches_drawer)

        assert expected == actual

        # Also check generator actually yields reported number of batches
        assert expected == len(list(samples_batches_drawer))

    def test_iterator_produces_deterministic_output_when_shuffling_is_turned_off(self):
        """
        Test that when shuffle option is set to False, two instaces of iterator based on same input
        data yield same results
        """

        # Set random seed - it shouldn't matter for this test, but just in case
        # there is a bug in code this way at least bug will be reproducible ^^
        random.seed(0)

        categories_samples_map = {
            1: np.arange(20),
            2: np.arange(15),
            3: np.arange(12),
            4: np.arange(20)
        }

        first_samples_batches_drawer = net.data.SamplesBatchesDrawer(
            categories_samples_map=categories_samples_map,
            categories_per_batch=2,
            samples_per_category=5,
            dataset_mode=net.constants.DatasetMode.VALIDATION
        )

        first_drawer_output = list(first_samples_batches_drawer)

        second_samples_batches_drawer = net.data.SamplesBatchesDrawer(
            categories_samples_map=categories_samples_map,
            categories_per_batch=2,
            samples_per_category=5,
            dataset_mode=net.constants.DatasetMode.VALIDATION
        )

        second_drawer_output = list(second_samples_batches_drawer)

        assert len(first_drawer_output) == 3
        assert first_drawer_output == second_drawer_output
