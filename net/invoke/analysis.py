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
    import tqdm

    import net.constants
    import net.data
    import net.ml
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    validation_data_loader = net.data.Cars196DataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.VALIDATION
    )

    validation_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(validation_data_loader),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None]))
    ).prefetch(32)

    all_embeddings = []
    all_labels = []

    data_iterator = iter(validation_dataset)

    prediction_model = tf.keras.models.load_model(
        filepath=config["model_dir"],
        compile=False,
        custom_objects={'average_ranking_position': net.ml.average_ranking_position})

    # Iterate over dataset to obtain embeddings and labels
    for _ in tqdm.tqdm(range(len(validation_data_loader))):

        images, labels = next(data_iterator)

        embeddings = prediction_model.predict(images)

        all_embeddings.extend(embeddings)
        all_labels.extend(labels)

    print(f"Embeddings len: {len(all_embeddings)}")
    print(f"Labels len: {len(all_labels)}")
