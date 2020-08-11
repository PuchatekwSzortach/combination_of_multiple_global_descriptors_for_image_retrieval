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

    import tensorflow as tf

    import net.constants
    import net.data
    import net.logging
    import net.ml
    import net.utilities

    config = net.utilities.read_yaml(config_path)

    training_data_loader = net.data.Cars196TrainingLoopDataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.TRAINING
    )

    validation_data_loader = net.data.Cars196TrainingLoopDataLoader(
        config=config,
        dataset_mode=net.constants.DatasetMode.VALIDATION
    )

    training_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(training_data_loader),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None]))
    ).prefetch(32)

    validation_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(validation_data_loader),
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, 224, 224, 3]), tf.TensorShape([None]))
    ).prefetch(32)

    model = net.ml.CGDImagesSimilarityComputer.get_model(
        image_size=config["image_size"])

    model.fit(
        x=training_dataset,
        epochs=200,
        steps_per_epoch=len(training_data_loader),
        validation_data=validation_dataset,
        validation_steps=len(validation_data_loader),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                monitor="val_average_ranking_position",
                filepath=config["model_dir"],
                save_best_only=True,
                save_weights_only=False,
                verbose=1),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_average_ranking_position",
                patience=15,
                verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_average_ranking_position",
                factor=0.1,
                patience=6,
                verbose=1),
            tf.keras.callbacks.CSVLogger(
                filename=config["training_metrics_log_path"]
            )
        ]
    )
