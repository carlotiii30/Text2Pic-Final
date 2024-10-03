import tensorflow as tf


def build_generator(latent_dim, vocab_size, max_length, embedding_dim=128):
    initializer = tf.keras.initializers.HeNormal()
    noise_input = tf.keras.Input(shape=(latent_dim,))
    caption_input = tf.keras.Input(shape=(max_length,))

    # Embedding y LSTM
    caption_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(
        caption_input
    )
    lstm_output = tf.keras.layers.LSTM(128, kernel_initializer=initializer)(
        caption_embedding
    )

    # Concatenar ruido y captions
    combined_input = tf.keras.layers.Concatenate()([noise_input, lstm_output])

    # Capas convolucionales transpuestas para generar una imagen de 32x32
    x = tf.keras.layers.Dense(
        128 * 4 * 4, activation="relu", kernel_initializer=initializer
    )(
        combined_input
    )  # Más filtros
    x = tf.keras.layers.Reshape((4, 4, 128))(x)

    x = tf.keras.layers.Conv2DTranspose(
        128,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=initializer,
    )(
        x
    )  # 8x8
    x = tf.keras.layers.Conv2DTranspose(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="relu",
        kernel_initializer=initializer,
    )(
        x
    )  # 16x16
    x = tf.keras.layers.Conv2DTranspose(
        3,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="tanh",
        kernel_initializer=initializer,
    )(
        x
    )  # 32x32

    generator = tf.keras.Model([noise_input, caption_input], x, name="generator")
    return generator


def build_discriminator(img_shape, vocab_size, max_length, embedding_dim=128):
    initializer = tf.keras.initializers.HeNormal()
    img_input = tf.keras.Input(shape=img_shape)

    # Capas convolucionales para procesar las imágenes
    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="leaky_relu",
        kernel_initializer=initializer,
    )(
        img_input
    )  # 16x16
    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="leaky_relu",
        kernel_initializer=initializer,
    )(
        x
    )  # 8x8
    x = tf.keras.layers.Flatten()(x)

    caption_input = tf.keras.Input(shape=(max_length,))
    caption_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(
        caption_input
    )
    caption_lstm = tf.keras.layers.LSTM(128, kernel_initializer=initializer)(
        caption_embedding
    )

    # Concatenar la imagen procesada y la caption embebida
    combined_input = tf.keras.layers.Concatenate()([x, caption_lstm])

    # Clasificación final
    combined_output = tf.keras.layers.Dense(1, kernel_initializer=initializer)(
        combined_input
    )  # Sin activación

    discriminator = tf.keras.Model(
        [img_input, caption_input], combined_output, name="discriminator"
    )
    return discriminator
