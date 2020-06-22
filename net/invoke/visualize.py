"""
Module with visualization related tasks
"""

import invoke


@invoke.task
def visualize_data(_context, config_path):
    """Visualize data

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

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.Cars196DataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.TRAINING
    )

    iterator = iter(training_data_loader)

    for _ in tqdm.tqdm(range(4)):

        categories_images_batch, categories_labels_batch = next(iterator)

        for images, labels in zip(categories_images_batch.values(), categories_labels_batch.values()):

            logger.info(
                vlogging.VisualRecord(
                    title="data",
                    imgs=[net.processing.ImageProcessor.get_denormalized_image(image) for image in images],
                    footnotes=labels
                )
            )
