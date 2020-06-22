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

    import tqdm

    import net.constants
    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.Cars196DataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.TRAINING
    )

    iterator = iter(training_data_loader)

    for _ in tqdm.tqdm(range(2)):

        categories_images_batch, categories_labels_batch = next(iterator)

        print(categories_images_batch.keys())
        print(categories_labels_batch.keys())
        print()
