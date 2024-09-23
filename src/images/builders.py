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
    Concatenate,
    BatchNormalization,
    Add,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


latent_dim = 150
embedding_dim = 300
vocab_size = 27549
max_length = 60
learning_rate = 0.0001


def build_models():
    # Construir el generador
    generator = _build_generator(vocab_size)

    # Construir el discriminador con las dimensiones de la imagen y captions
    discriminator = _build_discriminator(vocab_size)

    return generator, discriminator


def residual_block(x, filters, kernel_size=3, strides=1):
    res = Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    res = BatchNormalization()(res)
    res = LeakyReLU(0.2)(res)

    res = Conv2D(filters, kernel_size, strides=strides, padding="same")(res)
    res = BatchNormalization()(res)

    x = Add()([x, res])
    x = LeakyReLU(0.2)(x)
    return x


def _build_generator(vocab_size, embedding_dim=embedding_dim):
    noise_input = tf.keras.Input(shape=(latent_dim,))
    caption_input = tf.keras.Input(shape=(max_length,))

    # Embedding y LSTM para captions
    embedded_captions = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(
        caption_input
    )
    lstm_output = LSTM(256, return_sequences=False)(embedded_captions)

    # Concatenar el ruido y el output del LSTM
    concat = Concatenate()([noise_input, lstm_output])

    # Densas y reshape
    x = Dense(128 * 4 * 4, activation="relu", kernel_regularizer=l2(1e-4))(concat)
    x = BatchNormalization()(x)
    x = Reshape((4, 4, 128))(x)

    # Primera capa de Conv2DTranspose
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = residual_block(x, 128)

    # Segunda capa de Conv2DTranspose
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = residual_block(x, 128)

    # Tercera capa de Conv2DTranspose
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = residual_block(x, 64)

    # Cuarta capa de Conv2DTranspose con dimensiones no cuadradas
    x = Conv2DTranspose(64, kernel_size=(4, 5), strides=(1, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = residual_block(x, 64)

    # Capa de salida
    output = Conv2DTranspose(
        3, kernel_size=(4, 3), strides=(1, 1), padding="same", activation="tanh"
    )(x)

    model = Model([noise_input, caption_input], output)
    return model


def _build_discriminator(vocab_size, embedding_dim=embedding_dim):
    img_input = tf.keras.Input(shape=(32, 64, 3))
    caption_input = tf.keras.Input(shape=(max_length,))

    # Procesar la imagen
    x = Conv2D(64, kernel_size=4, strides=2, padding="same")(img_input)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.15)(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.15)(x)
    x = BatchNormalization()(x)  # Añadir BatchNormalization

    x = Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.15)(x)
    x = BatchNormalization()(x)  # Añadir BatchNormalization

    x = Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.15)(x)
    x = BatchNormalization()(x)  # Añadir BatchNormalization

    x = Flatten()(x)

    # Procesar las captions
    captions_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(
        caption_input
    )
    captions_flat = Flatten()(captions_embedding)

    # Concatenar imagen y captions
    combined = Concatenate()([x, captions_flat])

    # Capa densa final
    x = Dense(512, kernel_regularizer=l2(1e-4))(combined)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model([img_input, caption_input], x)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate, 0.5),
        metrics=["accuracy"],
    )

    return model


def build_conditional_gan(generator, discriminator):
    # Definir entradas
    noise_input = tf.keras.Input(shape=(latent_dim,))
    caption_input = tf.keras.Input(shape=(max_length,))

    # Generar imágenes a partir de las entradas
    img = generator([noise_input, caption_input])

    # Congelar el discriminador durante el entrenamiento del generador
    discriminator.trainable = False
    valid = discriminator([img, caption_input])

    combined = Model([noise_input, caption_input], valid)
    combined.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate, 0.5))

    return combined
