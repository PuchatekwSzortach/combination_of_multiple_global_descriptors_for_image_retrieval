"""
Module with logging utilities
"""

import os
import random

import cv2
import numpy as np
import vlogging

import net.analysis
import net.processing


class ImageRankingLogger:
    """
    Class for visualizing image ranking results
    """

    def __init__(self, logger, prediction_model):
        """
        Constructor

        :param logger: logging.Logger instaince
        :param prediction_model: keras model instaince
        """

        self.logger = logger
        self.prediction_model = prediction_model

    def log_ranking_on_batch(self, query_image, query_label, images, labels):
        """
        Log ranking result for query image againts all provided images

        :param query_image: 3D numpy array, image to query against
        :param query_label: int, label of query image
        :param images: list of 3D numpy arrays, images to rank w.r.t. query
        :param labels: list of ints, labels for all images
        """

        query_embedding = self.prediction_model.predict(np.array([query_image]))[0][0]
        embeddings = self.prediction_model.predict(images)[0]

        # Compute distances between query embedding and other embeddings
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)

        indices_sorted_by_distances = np.argsort(distances)

        ranked_images = [
            net.processing.ImageProcessor.get_denormalized_image(image)
            for image in images[indices_sorted_by_distances]]

        labels_sorted_by_distances = labels[indices_sorted_by_distances]

        # Draw a green frame around every image that has the same label as query image
        for image, label in zip(ranked_images, labels_sorted_by_distances):

            if label == query_label:

                cv2.circle(
                    img=image,
                    center=(127, 200),
                    radius=10,
                    color=(5, 220, 5),
                    thickness=-1
                )

        query_image = net.processing.ImageProcessor.get_denormalized_image(query_image)

        cv2.circle(
            img=query_image,
            center=(127, 200),
            radius=10,
            color=(255, 0, 0),
            thickness=-1
        )

        self.logger.info(
            vlogging.VisualRecord(
                title="query image",
                imgs=query_image,
                footnotes=f"label: {query_label}"
            )
        )

        self.logger.info(
            vlogging.VisualRecord(
                title="ranked images",
                imgs=ranked_images
            )
        )

        # Compute average position of images with same label as query label - the lower the number, the better
        # the ranking model
        self.logger.info("<h3>Average position of images with same label as query image: {:.3f}<h3></br><hr>".format(
            np.mean(np.where(labels_sorted_by_distances == query_label)[0])
        ))

    def log_ranking_on_dataset(self, data_loader, queries_count, logged_top_matches_count, image_size):
        """
        Log ranking results on a few random queries. For each query ranking is done across whole dataset.

        :param data_loader: net.data.Cars196AnalysisDataLoader instance
        :param queries_count: int, number of queries to run ranking on
        :param logged_top_matches_count: int, number of top matches to log for each query
        :param image_size: int, size to which images should be resized before logging
        """

        # Get embeddings and labels for whole dataset
        embeddings_matrix, labels_array = net.analysis.get_samples_embeddings(
            data_loader=data_loader,
            prediction_model=self.prediction_model,
            verbose=True)

        # Get images paths
        images_paths = \
            [os.path.join(data_loader.data_dir, annotation.filename) for annotation in data_loader.annotations]

        # Get indices of top k matched vectors for each vector
        top_k_indices_matrix = net.analysis.get_indices_of_k_most_similar_vectors(
            vectors=embeddings_matrix,
            k=logged_top_matches_count)

        import tqdm

        for _ in tqdm.tqdm(range(8)):

            ranking_data = []

            # For each query index - log query image and top matches
            for query_index in random.sample(population=range(len(labels_array)), k=queries_count):

                query_image = net.processing.ImageProcessor.get_resized_image(
                    image=cv2.imread(images_paths[query_index]),
                    target_size=image_size)

                query_image = cv2.circle(
                    img=query_image,
                    center=(127, 200),
                    radius=10,
                    color=(255, 0, 0),
                    thickness=-1
                )

                query_label = labels_array[query_index]

                matched_images = [
                    net.processing.ImageProcessor.get_resized_image(
                        image=cv2.imread(images_paths[match_index]),
                        target_size=image_size)
                    for match_index in top_k_indices_matrix[query_index]
                ]

                matched_images = [
                    cv2.circle(
                        img=image,
                        center=(127, 200),
                        radius=10,
                        color=(5, 220, 5),
                        thickness=-1
                    ) if labels_array[match_index] == query_label else image
                    for image, match_index in zip(matched_images, top_k_indices_matrix[query_index])
                ]

                ranking_data.append(
                    {
                        "query_image": query_image,
                        "matched_images": matched_images
                    }
                )

            self.logger.info(
                vlogging.VisualRecord(
                    title="ranking collage",
                    imgs=[self._get_ranking_images_collage(ranking_data, image_size)]
                )
            )

    def _get_ranking_images_collage(self, ranking_data, image_size):

        rows_count = len(ranking_data)
        columns_count = len(ranking_data[0]["matched_images"]) + 1

        # Draw all images onto one large image
        canvas = np.zeros(shape=(rows_count * image_size, columns_count * image_size, 3))

        for row_index in range(rows_count):

            canvas[row_index * image_size: (row_index + 1) * image_size, 0: image_size] = \
                ranking_data[row_index]["query_image"]

            for matching_image_index, matching_image in enumerate(ranking_data[row_index]["matched_images"]):

                canvas[
                    row_index * image_size: (row_index + 1) * image_size,
                    (matching_image_index + 1) * image_size: (matching_image_index + 2) * image_size
                ] = matching_image

        return canvas
