import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Reshape,
    Embedding,
    LSTM,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    Dropout,
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

latent_dim = 80
vocab_size = 27549
max_length = 60


def build_models():
    generator = _build_generator(vocab_size)
    discriminator = _build_discriminator((32, 64, 3))  # Entrada del discriminador
    return generator, discriminator


def _build_generator(vocab_size, embedding_dim=100):
    model = Sequential()
    model.add(tf.keras.Input(shape=(128,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dense(128 * 4 * 4, activation="relu"))  # Ajustado para (4, 4)
    model.add(Reshape((4, 4, 128)))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Conv2DTranspose(64, kernel_size=(4, 5), strides=(1, 2), padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(
        Conv2DTranspose(
            3, kernel_size=(4, 3), strides=(1, 1), padding="same", activation="tanh"
        )
    )

    return model


def _build_discriminator(input_shape=(32, 64, 3)):  # Nueva resolución
    model = Sequential()
    model.add(
        Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=input_shape)
    )
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model


def build_conditional_gan(generator, discriminator):
    discriminator.compile(
        loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5), metrics=["accuracy"]
    )

    z = tf.keras.Input(shape=(128,))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.5))

    return combined
