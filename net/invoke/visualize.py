"""
Module with visualization related tasks
"""

import invoke


@invoke.task
def visualize_data(_context, config_path):
    """
    Visualize data

    :param _context: context
    :type _context: invoke.Context
    :param config_path: path to configuration file
    :type config_path: str
    """

    import tqdm
    import vlogging

    import net.data
    import net.processing
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    logger = net.utilities.get_logger(
        path=config["log_path"]
    )

    data_loader = net.data.Cars196TrainingLoopDataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.TRAINING
    )

    iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        images_batch, labels_batch = next(iterator)

        logger.info(
            vlogging.VisualRecord(
                title="data",
                imgs=[net.processing.ImageProcessor.get_denormalized_image(image) for image in images_batch],
                footnotes=str(labels_batch)
            )
        )


@invoke.task
def visualize_predictions(_context, config_path):
    """
    Visualize image similarity ranking predictions

    :param _context: invoke.Context instance
    :param config_path: str, path to configuration file
    """

    import random

    import tensorflow as tf
    import tqdm

    import net.data
    import net.logging
    import net.ml
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    logger = net.utilities.get_logger(
        path=config["log_path"]
    )

    data_loader = net.data.Cars196TrainingLoopDataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.VALIDATION
    )

    prediction_model = tf.keras.models.load_model(
        filepath=config["model_dir"],
        compile=False,
        custom_objects={'average_ranking_position': net.ml.average_ranking_position})

    image_ranking_logger = net.logging.ImageRankingLogger(
        logger=logger,
        prediction_model=prediction_model
    )

    data_iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        images, labels = next(data_iterator)

        query_index = random.choice(range(len(images)))

        image_ranking_logger.log_ranking(
            query_image=images[query_index],
            query_label=labels[query_index],
            images=images,
            labels=labels
        )
