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

    for epoch_index in tqdm.tqdm(range(20)):

        similarity_computer.model.fit(
            x=iter(training_data_loader),
            epochs=1,
            steps_per_epoch=len(training_data_loader)
        )

        embeddings = similarity_computer.model.predict(test_images)

        print("Small embeddings section")
        print(embeddings[5:8, 5:8])
        print()

        if epoch_index % 3 == 0:

            logger.info(f"<h1>Epoch {epoch_index}</h1>")

            image_ranking_logger.log_ranking(
                query_image=test_images[query_index],
                query_label=test_labels[query_index],
                images=test_images,
                labels=test_labels
            )
