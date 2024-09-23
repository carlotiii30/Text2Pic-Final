import os
import numpy as np
import pickle

from src.images import builders, dataset, text_process, training
from src import utils

from PIL import Image


def generate_and_save_image(generator, text_sequence, filename="generated_image.png"):
    noise = np.random.normal(0, 1, (1, builders.latent_dim))
    text_sequence = np.array(text_sequence).reshape(1, -1)
    generated_image = generator.predict([noise, text_sequence])
    generated_image = generated_image.squeeze()

    # Asegúrate de que la imagen generada tenga tres canales
    if len(generated_image.shape) == 2:
        generated_image = np.stack((generated_image,) * 3, axis=-1)

    flattened_image = generated_image.flatten()

    # Desnormalizar los píxeles
    pixels_denormalized = [denormalize_pixel(pixel) for pixel in flattened_image]

    # Convertimos la lista de píxeles en un array de numpy
    image_data = np.array(pixels_denormalized, dtype=np.uint8)

    # Redimensionamos la imagen para que sea 3D (alto, ancho, canales)
    image_data = image_data.reshape(
        (generated_image.shape[0], generated_image.shape[1], 3)
    )

    # Creamos la imagen con PIL
    img = Image.fromarray(image_data, "RGB")  # 'RGB' para imágenes en color

    # Guardamos la imagen en un fichero
    img.save(filename)


def denormalize_pixel(pixel):
    return int((pixel + 1) * 127.5)


# Cargar dataset COCO
data_dir = "data/coco"
annotation_file = os.path.join(data_dir, "annotations/captions_train2017.json")
train_dir = os.path.join(data_dir, "train2017")
subset = dataset.load_coco_subset(train_dir, annotation_file)
# full_dataset = dataset.load_coco_dataset(train_dir, annotation_file)

# Construir modelos
generator, discriminator = builders.build_models()
combined = builders.build_conditional_gan(generator, discriminator)

# Entrenar modelos
# training.train(generator, discriminator, combined, subset)

# Guardar los modelos
# utils.save_model_weights(generator, "data/models/generator_weights_a.weights.h5")
# utils.save_model_weights(
#     discriminator, "data/models/discriminator_weights_a.weights.h5"
# )
# utils.save_model_weights(combined, "data/models/combined_weights_a.weights.h5")

# Cargar los modelos
generator = utils.load_model_with_weights(
    generator, "data/models/generator_weights_reentrenado_0.weights.h5"
)
discriminator = utils.load_model_with_weights(
    discriminator, "data/models/discriminator_weights_reentrenado_0.weights.h5"
)
combined = utils.load_model_with_weights(
    combined, "data/models/combined_weights_reentrenado_0.weights.h5"
)

# Entrenamiento en bucle
for i in range(7):
    training.train(generator, discriminator, combined, subset)
    utils.save_model_weights(
        generator, f"data/models/generator_weights_reentrenado_{i}.weights.h5"
    )
    utils.save_model_weights(
        discriminator, f"data/models/discriminator_weights_reentrenado_{i}.weights.h5"
    )
    utils.save_model_weights(
        combined, f"data/models/combined_weights_reentrenado_{i}.weights.h5"
    )

# Tokenizer
tokenizer_path = "data/tokenizer.pkl"

with open(tokenizer_path, "rb") as file:
    tokenizer = pickle.load(file)

# Preprocesar el texto de entrada
input_text = "A boy standing on a beach"
text_sequence = dataset.preprocess_text(input_text, tokenizer, text_process.max_length)

# Generar y visualizar la imagen
generate_and_save_image(generator, text_sequence)
