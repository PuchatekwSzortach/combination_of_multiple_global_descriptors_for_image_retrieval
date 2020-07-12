"""
Module with logging utilities
"""

import cv2
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

    def log_ranking(self, query_image, query_label, images, labels):
        """
        Log ranking result for query image againts all provided images

        :param query_image: 3D numpy array, image to query against
        :param query_label: int, label of query image
        :param images: list of 3D numpy arrays, images to rank w.r.t. query
        :param labels: list of ints, labels for all images
        """

        query_embedding = self.prediction_model.predict(np.array([query_image]))[0]
        embeddings = self.prediction_model.predict(images)

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

                cv2.rectangle(
                    img=image,
                    pt1=(0, 0),
                    pt2=(image.shape[1], image.shape[0]),
                    color=(0, 255, 0),
                    thickness=8
                )

        self.logger.info(
            vlogging.VisualRecord(
                title="query image",
                imgs=[net.processing.ImageProcessor.get_denormalized_image(query_image)],
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
