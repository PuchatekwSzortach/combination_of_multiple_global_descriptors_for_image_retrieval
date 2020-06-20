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

    import net.constants
    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.Cars196DataLoader(
        data_dir=config["data_dir"],
        annotations_path=config["annotations_path"],
        dataset_mode=net.constants.DatasetMode.TRAINING
    )

    print(training_data_loader)
    print(len(training_data_loader.annotations))
