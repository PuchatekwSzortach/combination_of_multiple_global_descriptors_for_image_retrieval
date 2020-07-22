"""
Module with machine learning tasks
"""

import invoke


@invoke.task
def train(_context, config_path):
    """
    Train model

    :param _context: invoke.Context instance
    :param config_path: str, path to configuration file
    """

    import random

    import net.constants
    import net.data
    import net.logging
    import net.ml
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.Cars196DataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.TRAINING
    )

    validation_data_loader = net.data.Cars196DataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.VALIDATION
    )

    validation_data_iterator = iter(validation_data_loader)

    test_images, test_labels = next(validation_data_iterator)
    query_index = random.choice(range(len(test_images)))

    logger = net.utilities.get_logger(path=config["log_path"])

    similarity_computer = net.ml.ImagesSimilarityComputer()

    image_ranking_logger = net.logging.ImageRankingLogger(
        logger=logger,
        prediction_model=similarity_computer.model
    )

    for epoch_index in range(50):

        print(f"Epoch {epoch_index}")

        similarity_computer.model.fit(
            x=iter(training_data_loader),
            epochs=1,
            steps_per_epoch=len(training_data_loader)
        )

        if epoch_index % 2 == 0:

            logger.info(f"<h1>Epoch {epoch_index}</h1>")

            image_ranking_logger.log_ranking(
                query_image=test_images[query_index],
                query_label=test_labels[query_index],
                images=test_images,
                labels=test_labels
            )
