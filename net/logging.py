"""
Module with logging utilities
"""

import numpy as np

import vlogging

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

    def log_ranking(self, query_image, images):
        """
        Log ranking result for query image againts all provided images

        :param query_image: 3D numpy array, image to query against
        :param images: list of 3D numpy arrays, images to rank w.r.t. query
        """

        query_embedding = self.prediction_model.predict(np.array([query_image]))[0]
        embeddings = self.prediction_model.predict(images)

        # Compute distances between query embedding and other embeddings
        distances = np.linalg.norm(embeddings - query_embedding, axis=1)

        indices_sorted_by_distances = np.argsort(distances)

        ranked_images = images[indices_sorted_by_distances]

        self.logger.info(
            vlogging.VisualRecord(
                title="query image",
                imgs=[net.processing.ImageProcessor.get_denormalized_image(query_image)]
            )
        )

        self.logger.info(
            vlogging.VisualRecord(
                title="ranked images",
                imgs=[net.processing.ImageProcessor.get_denormalized_image(image) for image in ranked_images]
            )
        )
