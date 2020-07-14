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

    import tqdm

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

    similarity_computer = net.ml.ImagesSimilarityComputer()

    data_iterator = iter(training_data_loader)
    test_images, test_labels = next(data_iterator)
    query_index = random.choice(range(len(test_images)))

    logger = net.utilities.get_logger(path=config["log_path"])

    image_ranking_logger = net.logging.ImageRankingLogger(
        logger=logger,
        prediction_model=similarity_computer.model
    )

    # print(f"Labels are: {test_labels}")

    # embeddings = similarity_computer.model.predict(test_images)

    # loss = net.ml.get_hard_aware_point_to_set_loss_op(
    #     labels=test_labels,
    #     embeddings=embeddings
    # )

    # print(loss)

    for epoch_index in tqdm.tqdm(range(20)):

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
