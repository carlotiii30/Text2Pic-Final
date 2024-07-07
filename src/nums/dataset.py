import keras
import numpy as np
import tensorflow as tf

batch_size = 64
latent_dim = 128


def load_dataset():
    pre_processed_dataset = keras.datasets.mnist, 10, 1, 28
    dataset, num_classes, num_channels, image_size = pre_processed_dataset
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    all_images = np.concatenate([x_train, x_test]).astype("float32") / 255.0
    all_labels = keras.utils.to_categorical(
        np.concatenate([y_train, y_test]), num_classes
    )

    all_images = np.reshape(all_images, (-1, image_size, image_size, num_channels))

    dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    return dataset
