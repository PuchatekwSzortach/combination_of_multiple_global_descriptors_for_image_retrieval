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

    import os

    import net.constants
    import net.data
    import net.ml
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.Cars196DataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.TRAINING
    )

    similarity_computer = net.ml.ImagesSimilarityComputer()

    similarity_computer.model.fit(
        x=iter(training_data_loader),
        epochs=5,
        steps_per_epoch=len(training_data_loader)
    )

    os.makedirs(config["model_dir"], exist_ok=True)

    similarity_computer.model.save(
        filepath=config["model_dir"]
    )
