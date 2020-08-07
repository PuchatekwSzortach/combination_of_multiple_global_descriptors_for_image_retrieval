"""
Module with analysis tasks
"""

import invoke


@invoke.task
def analyze_model_performance(_context, config_path):
    """
    Analyze model performance

    :param _context: invoke.Context instance
    :param config_path: str, path to configuration file
    """

    import tensorflow as tf

    import net.analysis
    import net.constants
    import net.data
    import net.ml
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    validation_data_loader = net.data.Cars196DataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.VALIDATION
    )

    prediction_model = tf.keras.models.load_model(
        filepath=config["model_dir"],
        compile=False,
        custom_objects={'average_ranking_position': net.ml.average_ranking_position})

    embeddings_matrix, labels_array = net.analysis.get_samples_embeddings(
        data_loader=validation_data_loader,
        prediction_model=prediction_model,
        verbose=True)

    for k in [1, 2, 4, 8]:

        score = net.analysis.get_recall_at_k_score(
            vectors=embeddings_matrix,
            labels=labels_array,
            k=k
        )

        print(f"Recall at {k} is: {score:.3f}")
