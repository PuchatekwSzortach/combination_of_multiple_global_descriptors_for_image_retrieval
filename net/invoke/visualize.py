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

    import net.utilities

    config = net.utilities.read_yaml(config_path)

    logger = net.utilities.get_logger(
        path=config["log_path"]
    )

    logger.info("Some data here")
