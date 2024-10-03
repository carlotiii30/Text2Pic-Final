import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from src.images import builders, dataset, training


def generate_and_display_image(i, generator, tokenizer, caption, latent_dim=100):
    # Generar un vector de ruido aleatorio
    noise = tf.random.normal([1, latent_dim])

    # Tokenizar la caption de prueba
    caption_sequence = tokenizer.texts_to_sequences([caption])
    caption_padded = tf.keras.preprocessing.sequence.pad_sequences(
        caption_sequence, maxlen=60, padding="post"
    )

    # Generar la imagen n el modelo generador
    generated_image = generator.predict([noise, caption_padded])

    # Remover la normalización para visualizar correctamente la imagen (valores entre 0 y 1)
    generated_image = (
        generated_image + 1
    ) / 2.0  # Si usaste tanh en la salida del generador

    # Guardar la imagen generada
    plt.imsave(f"generated_image_{i}.png", generated_image[0])


# Establecer el uso de Mixed Precision para mejorar el rendimiento
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

strategy = tf.distribute.MirroredStrategy()

# Parámetros
latent_dim = 100
image_shape = (32, 32, 3)  # Tamaño de las imágenes durante el entrenamiento
vocab_size = 10000  # Vocabulario máximo de las captions
max_length = 60  # Longitud máxima de las captions

# Cargar dataset COCO
data_dir = "data/coco"
annotation_file = os.path.join(data_dir, "annotations/captions_train2017.json")
train_dir = os.path.join(data_dir, "train2017")

image_id_to_caption = dataset.load_coco_annotations(annotation_file)
image_paths, captions = dataset.load_image_caption_pairs(
    train_dir, image_id_to_caption, subset_size=1500
)
fullset, tokenizer = dataset.load_coco_dataset(image_paths, captions)

# Construir el generador y discriminador
generator = builders.build_generator(latent_dim, vocab_size, max_length)
discriminator = builders.build_discriminator(image_shape, vocab_size, max_length)

# Cargar los modelos
# generator = tf.keras.models.load_model("data/models/generator_final.h5")
# discriminator = tf.keras.models.load_model("data/models/discriminator_final.h5")

# # Ajustar las tasas de aprendizaje para el reentrenamiento
# optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.0002)
# optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.00005)

# Caption de prueba
caption = "a man next to an ambulance"

for i in range(6):
    print(f"Epoch {i + 1}")
    training.train_gan(generator, discriminator, fullset, latent_dim, epochs=9)
    generate_and_display_image(i, generator, tokenizer, caption, latent_dim=100)
