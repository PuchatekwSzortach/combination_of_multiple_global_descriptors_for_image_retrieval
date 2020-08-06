"""
Module with logging utilities
"""

import random

import cv2
import numpy as np
import tensorflow as tf
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


class LoggingCallback(tf.keras.callbacks.Callback):
    """
    Callback that logs prediction results
    """

    def __init__(self, logger, model, data_loader):
        """
        Constructor

        :param logger: logging.Logger instance
        :param model: keras.Model instance
        :param data_loader: data loader instance
        """

        super().__init__()

        self.logger = logger

        self.image_ranking_logger = ImageRankingLogger(
            logger=logger,
            prediction_model=model
        )

        data_iterator = iter(data_loader)

        self.test_images, self.test_labels = next(data_iterator)
        self.query_index = random.choice(range(len(self.test_images)))

    def on_epoch_end(self, epoch, logs=None):

        self.logger.info(f"<h1>Epoch {epoch}</h1>")

        self.image_ranking_logger.log_ranking(
            query_image=self.test_images[self.query_index],
            query_label=self.test_labels[self.query_index],
            images=self.test_images,
            labels=self.test_labels
        )
