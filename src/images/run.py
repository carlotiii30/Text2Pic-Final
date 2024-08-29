import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

from src.images import builders, dataset, text_process, training
from src import utils


def generate_and_visualize_image(generator, text_sequence):
    noise = np.random.normal(0, 1, (1, builders.latent_dim))
    combined_input = np.hstack((noise, text_sequence))
    generated_image = generator.predict(combined_input)
    generated_image = generated_image.squeeze()
    plt.figure()
    plt.imshow((generated_image * 127.5 + 127.5).astype(np.uint8))
    plt.axis("off")
    plt.show()


# Cargar dataset COCO
data_dir = "data/coco"
annotation_file = os.path.join(data_dir, "annotations/captions_train2017.json")
train_dir = os.path.join(data_dir, "train2017")
subset = dataset.load_coco_subset(train_dir, annotation_file)
# full_dataset = dataset.load_coco_dataset(train_dir, annotation_file)

# Construir modelos
generator, discriminator = builders.build_models()
generator.summary()
discriminator.summary()
combined = builders.build_conditional_gan(generator, discriminator)
combined.summary()

# # Entrenar modelos
# training.train(generator, discriminator, combined, subset)

# # Guardar los modelos
# utils.save_model_weights(generator, "data/models/generator_weights.weights.h5")
# utils.save_model_weights(discriminator, "data/models/discriminator_weights.weights.h5")
# utils.save_model_weights(combined, "data/models/combined_weights.weights.h5")

# Cargar los modelos
generator = utils.load_model_with_weights(
    generator, "data/models/generator_weights.weights.h5"
)
discriminator = utils.load_model_with_weights(
    discriminator, "data/models/discriminator_weights.weights.h5"
)
combined = utils.load_model_with_weights(
    combined, "data/models/combined_weights.weights.h5"
)

# Tokenizer
tokenizer_path = "data/tokenizer.pkl"

with open(tokenizer_path, "rb") as file:
    tokenizer = pickle.load(file)

# # Preprocesar el texto de entrada
input_text = "A giraffe"
text_sequence = dataset.preprocess_text(input_text, tokenizer, text_process.max_length)

# # Generar y visualizar la imagen
generate_and_visualize_image(generator, text_sequence)
