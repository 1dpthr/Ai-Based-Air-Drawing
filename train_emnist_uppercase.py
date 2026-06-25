"""
Train CNN on EMNIST uppercase letters (A–Z) for air drawing recognition.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D

DATASET_NAME = "emnist/letters"
DIGITS_DATASET = "emnist/digits"
NUM_CLASSES = 36  # 0-9 + A-Z
IMAGE_SHAPE = (28, 28, 1)
BATCH_SIZE = 128
EPOCHS = 15
VALIDATION_SPLIT = 0.1
MODEL_PATH = "emnist_uppercase_model.h5"
RANDOM_SEED = 42
CLASS_NAMES = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]


def fix_emnist_orientation(image: tf.Tensor) -> tf.Tensor:
    image = tf.transpose(image)
    image = tf.reverse(image, axis=[1])
    return image


def preprocess_digit(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocess EMNIST digits (labels 0-9)."""
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, (28, 28))
    image = fix_emnist_orientation(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=-1)
    # Digits are already 0-9, no offset needed
    label = tf.cast(label, tf.int32)
    return image, label


def thicken_strokes(image: tf.Tensor) -> tf.Tensor:
    """Simulate thicker air-drawn lines (erosion on white-bg dark letter)."""
    inverted = 1.0 - image
    k = 3
    inverted = tf.nn.max_pool2d(inverted[None, ...], ksize=k, strides=1, padding="SAME")[0]
    return 1.0 - inverted


def augment_train(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.pad_to_bounding_box(image, 2, 2, 32, 32)
    image = tf.image.random_crop(image, size=[28, 28, 1])
    if tf.random.uniform([]) > 0.5:
        image = thicken_strokes(image)
    return image, label


def preprocess_letter(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocess EMNIST letters (labels 1-26, offset to 10-35)."""
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, (28, 28))
    image = fix_emnist_orientation(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=-1)
    # Letters are labeled 1-26, map to 10-35 (after digits 0-9)
    label = tf.cast(label - 1 + 10, tf.int32)
    return image, label


def load_datasets():
    print("Loading EMNIST digits and letters …")
    (ds_digits_train, ds_digits_test), digits_info = tfds.load(
        DIGITS_DATASET,
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )
    (ds_letters_train, ds_letters_test), letters_info = tfds.load(
        DATASET_NAME,
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )

    # Preprocess each dataset with appropriate label mapping
    ds_digits_train = ds_digits_train.map(preprocess_digit, num_parallel_calls=tf.data.AUTOTUNE)
    ds_digits_test = ds_digits_test.map(preprocess_digit, num_parallel_calls=tf.data.AUTOTUNE)
    ds_letters_train = ds_letters_train.map(preprocess_letter, num_parallel_calls=tf.data.AUTOTUNE)
    ds_letters_test = ds_letters_test.map(preprocess_letter, num_parallel_calls=tf.data.AUTOTUNE)

    # Combine datasets
    ds_train = ds_digits_train.concatenate(ds_letters_train)
    ds_test = ds_digits_test.concatenate(ds_letters_test)

    num_train = digits_info.splits["train"].num_examples + letters_info.splits["train"].num_examples
    val_count = int(num_train * VALIDATION_SPLIT)

    print(f"  Total Train: {num_train - val_count}  Val: {val_count}  Test: {ds_test.cardinality().numpy()}")

    ds_shuffled = ds_train.shuffle(50_000, seed=RANDOM_SEED, reshuffle_each_iteration=False)

    val_ds = ds_shuffled.take(val_count).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_ds = (
        ds_shuffled.skip(val_count)
        .map(augment_train, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds


def build_model() -> Sequential:
    return Sequential(
        [
            Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=IMAGE_SHAPE),
            BatchNormalization(),
            Conv2D(32, (3, 3), padding="same", activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.2),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="emnist_uppercase_cnn",
    )


def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_ds, val_ds, test_ds = load_datasets()
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    print("\nTraining …")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

    model.save(MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
