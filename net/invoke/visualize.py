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
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    logger = net.utilities.get_logger(
        path=config["log_path"]
    )

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.Cars196DataLoader(
        data_dir=config["data_dir"],
        annotations_path=config["annotations_path"],
        dataset_mode=net.constants.DatasetMode.TRAINING,
        categories_per_batch=config["train"]["categories_per_batch"],
        samples_per_category=config["train"]["samples_per_category"]
    )

    iterator = iter(training_data_loader)

    for _ in tqdm.tqdm(range(4)):

        categories_images_batch, categories_labels_batch = next(iterator)

        for images, labels in zip(categories_images_batch.values(), categories_labels_batch.values()):

            logger.info(
                vlogging.VisualRecord(
                    title="batch",
                    imgs=images,
                    footnotes=labels
                )
            )
