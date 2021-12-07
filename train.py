import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def train_model(directory):
    cwd = os.getcwd()
    data_dir = f"{cwd}/{directory}/train"
    project = f"{cwd}/{directory}"

    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1.0 / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    model = Sequential(
        [
            layers.experimental.preprocessing.Rescaling(
                1.0 / 255, input_shape=(img_height, img_width, 3)
            ),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    epochs = 30
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(f"{project}/disclosure-model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help train the model")
    parser.add_argument(
        "--dir",
        default="combined",
        help="Where is the dataset",
    )

    args = parser.parse_args()
    train_model(args.dir)
