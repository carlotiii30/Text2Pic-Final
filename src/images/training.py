import numpy as np
from tensorflow.image import resize

latent_dim = 80
max_length = 60


def resize_real_images(imgs):
    return resize(imgs, (32, 64))  # Cambiado de (64, 128) a (32, 64)


def train(
    generator,
    discriminator,
    combined,
    dataset,
    epochs=1,
    batch_size=64,
):
    print("Iniciando el entrenamiento...")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} iniciada.")

        for batch_idx, batch in enumerate(dataset, start=1):
            imgs, captions = batch
            resized_imgs = resize_real_images(imgs)

            # Verifica que el tamaño del batch sea correcto
            if imgs.shape[0] != batch_size:
                print(f"Batch {batch_idx} ignorado debido a tamaño incorrecto.")
                continue

            noise = np.random.normal(0, 1, (batch_size, 68))  # Ajusta según necesidad
            captions_input = np.array(captions)
            combined_input = np.hstack((noise, captions_input))

            # Genera imágenes falsas
            gen_imgs = generator.predict(combined_input)

            # Etiquetas para las imágenes reales y falsas
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Entrena al discriminador (real y falso)
            d_loss_real = discriminator.train_on_batch(resized_imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Entrena al generador
            discriminator.trainable = False
            combined_input = combined_input.reshape(
                (batch_size, 128)
            )  # Ajusta si es necesario
            g_loss = combined.train_on_batch(combined_input, valid)
            discriminator.trainable = True

            print(f"Batch {batch_idx} procesado. d_loss: {d_loss}, g_loss: {g_loss}")

        print(f"Epoch {epoch+1}/{epochs} FINALIZADA")

    print("Entrenamiento completado.")
