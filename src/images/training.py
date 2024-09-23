from matplotlib import pyplot as plt
import numpy as np
from tensorflow.image import resize
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

latent_dim = 150
max_length = 60  # Longitud máxima de los captions


def resize_real_images(imgs):
    return resize(imgs, (32, 64))


def train(
    generator,
    discriminator,
    combined,
    dataset,
    epochs=5,
    batch_size=64,
    patience=100,
):
    print("Iniciando el entrenamiento...")

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="g_loss", patience=patience, restore_best_weights=True, mode="min"
    )
    early_stopping.set_model(combined)

    lr_scheduler = ReduceLROnPlateau(
        monitor="g_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )
    lr_scheduler.set_model(combined)

    g_losses = []
    d_losses = []

    with open("training_log.txt", "w") as file:
        file.write("Epoch\tGenerator Loss\tDiscriminator Loss\n")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs} iniciada.")

            for batch in dataset:
                imgs, captions = batch
                resized_imgs = resize_real_images(imgs)

                if imgs.shape[0] != batch_size:
                    continue

                # Ajusta el tamaño del vector de ruido
                noise_dim = latent_dim
                noise = np.random.normal(0, 1, (batch_size, noise_dim))
                captions_input = np.array(captions)

                # Entrenar al discriminador
                for _ in range(2):
                    # Generar imágenes falsas
                    gen_imgs = generator.predict([noise, captions_input])

                    # Etiquetas para las imágenes reales y falsas
                    valid = np.ones((batch_size, 1)) * 0.9
                    fake = np.zeros((batch_size, 1)) + 0.1

                    # Entrena al discriminador (real y falso)
                    d_loss_real = discriminator.train_on_batch(
                        [resized_imgs, captions_input], valid
                    )
                    d_loss_fake = discriminator.train_on_batch(
                        [gen_imgs, captions_input], fake
                    )

                    # Calcula la pérdida del discriminador
                    d_losses.append(0.5 * np.add(d_loss_real, d_loss_fake))

                # Entrena al generador
                discriminator.trainable = False
                g_loss = combined.train_on_batch([noise, captions_input], valid)
                discriminator.trainable = True

                # Guarda la pérdida del generador
                g_losses.append(g_loss)

            # Guardar el estado del modelo y aplicar early stopping
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            print(
                f"Epoch {epoch+1}/{epochs} FINALIZADA con g_loss: {avg_g_loss} y d_loss: {avg_d_loss}"
            )

            file.write(f"{epoch + 1}\t{avg_g_loss}\t{avg_d_loss}\n")

            # Callbacks
            early_stopping.on_epoch_end(epoch, {"g_loss": avg_g_loss})
            lr_scheduler.on_epoch_end(epoch, {"g_loss": avg_g_loss})

            # Aplicar early stopping si no hay mejoras
            if early_stopping.stopped_epoch:
                print(
                    f"Entrenamiento detenido temprano en la epoch {early_stopping.stopped_epoch + 1}"
                )
                break

        # Graficar las pérdidas al finalizar el entrenamiento
        # plt.figure(figsize=(10, 5))
        # plt.plot(g_losses, label="Generator Loss", color="blue")
        # plt.plot(d_losses, label="Discriminator Loss", color="red")
        # plt.title("Losses during Training")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.grid()
        # plt.show()

    print("Entrenamiento completado.")
