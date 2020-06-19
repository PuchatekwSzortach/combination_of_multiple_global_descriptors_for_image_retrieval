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

    import os

    import cv2
    import scipy.io
    import tqdm
    import vlogging

    import net.data
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    logger = net.utilities.get_logger(
        path=config["log_path"]
    )

    annotations_data = scipy.io.loadmat("/data/cars_196/cars_annos.mat")

    annotations = annotations_data["annotations"].flatten()
    categories_names = annotations_data["class_names"].flatten()

    for annotation_matrix in tqdm.tqdm(annotations[:10]):

        annotation = net.data.Cars196Annotation(annotation_matrix=annotation_matrix)

        logger.info(vlogging.VisualRecord(
            title=categories_names[annotation.category_id][0],
            imgs=[cv2.imread(os.path.join(config["data_dir"], annotation.filename))]))
