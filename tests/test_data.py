"""
Tests for net.data module
"""

import collections
import random

import numpy as np

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
            samples_per_category=5
        )

        # For each category we should be able to draw 4 samples, and we draw 2 categories from total of 4 categories,
        # thus expected to have a total of 8 possible batches
        expected = 8
        actual = len(samples_batches_drawer)

        assert expected == actual

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
            samples_per_category=5
        )

        # category "3" has 12 samples, so we should be able to take only 2 batches from it.
        # Since we draw 2 categories out of 4 per batch, we then expect to have 2 * 2 = 4 total batches
        expected = 4
        actual = len(samples_batches_drawer)

        assert expected == actual

    def test_iterator(self):
        """
        Test iterator yields samples as expected
        """

        # Set random seed
        random.seed(0)

        categories_samples_map = {
            1: np.arange(20),
            2: np.arange(15),
            3: np.arange(12),
            4: np.arange(20)
        }

        samples_batches_drawer = net.data.SamplesBatchesDrawer(
            categories_samples_map=categories_samples_map,
            categories_per_batch=2,
            samples_per_category=5
        )

        categories_drawn_samples_map = collections.defaultdict(list)

        expected_batches_count = 4

        # Assert batcher drawer reports expected length
        assert len(samples_batches_drawer) == expected_batches_count

        batches_count = 0

        for batch in samples_batches_drawer:

            batches_count += 1

            for category, samples in batch.items():

                categories_drawn_samples_map[category].extend(samples)

        # Assert batches drawer returned expected number of batches
        assert batches_count == expected_batches_count

        for category, drawn_samples in categories_drawn_samples_map.items():

            # Assert drawn samples all come from original samples for category,
            # and are unique (which they should be given original data has unique elements only)
            assert set(categories_samples_map[category]).issuperset(drawn_samples)
            assert len(drawn_samples) == len(set(drawn_samples))
