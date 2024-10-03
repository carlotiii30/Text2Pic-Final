import os
import tensorflow as tf
import matplotlib.pyplot as plt


def wasserstein_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)


def train_gan(
    generator,
    discriminator,
    dataset,
    latent_dim,
    epochs=9,
    save_every=3,
    save_path="data/models",
):
    batch_size = 16
    valid = tf.random.uniform(
        (batch_size, 1), 0.9, 1.1
    )  # Menos variación en etiquetas reales
    fake = tf.random.uniform(
        (batch_size, 1), 0.0, 0.1
    )  # Menos ruido en etiquetas falsas

    # Tasas de aprendizaje ajustadas
    optimizer_g = tf.keras.optimizers.Adam(
        learning_rate=0.0003
    )  # Menor tasa para estabilizar
    optimizer_d = tf.keras.optimizers.Adam(
        learning_rate=0.00002
    )  # Menor tasa para estabilizar

    discriminator.compile(loss=wasserstein_loss, optimizer=optimizer_d)

    noise = tf.keras.Input(shape=(latent_dim,))
    caption = tf.keras.Input(shape=(dataset.element_spec[1].shape[1],))
    generated_image = generator([noise, caption])

    discriminator.trainable = False
    valid_prediction = discriminator([generated_image, caption])

    gan_model = tf.keras.Model([noise, caption], valid_prediction)
    gan_model.compile(loss=wasserstein_loss, optimizer=optimizer_g)

    g_losses_history = []
    d_losses_history = []

    for epoch in range(epochs):
        g_losses = []
        d_losses = []

        for image_batch, caption_batch in dataset:
            batch_size = image_batch.shape[0]
            noise = tf.random.normal([batch_size, latent_dim])

            generated_images = generator.predict([noise, caption_batch])

            # Entrenar el discriminador
            d_loss_real = discriminator.train_on_batch(
                [image_batch, caption_batch], valid[:batch_size]
            )
            d_loss_fake = discriminator.train_on_batch(
                [generated_images, caption_batch], fake[:batch_size]
            )
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_losses.append(d_loss)

            for _ in range(5):
                g_loss = gan_model.train_on_batch(
                    [noise, caption_batch], valid[:batch_size]
                )
                g_losses.append(g_loss)

        avg_g_loss = tf.reduce_mean(g_losses).numpy()
        avg_d_loss = tf.reduce_mean(d_losses).numpy()

        g_losses_history.append(avg_g_loss)
        d_losses_history.append(avg_d_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - D Loss: {avg_d_loss:.4f} - G Loss: {avg_g_loss:.4f}"
        )

        with open("training.log", "a") as f:
            f.write(
                f"Epoch {epoch + 1}/{epochs} - D Loss: {avg_d_loss:.4f} - G Loss: {avg_g_loss:.4f}\n"
            )

        if (epoch + 1) % save_every == 0:
            generator.save(os.path.join(save_path, f"generator_epoch_{epoch + 1}.h5"))
            discriminator.save(
                os.path.join(save_path, f"discriminator_epoch_{epoch + 1}.h5")
            )
            print(f"Modelos guardados en la época {epoch + 1}.")

    generator.save(os.path.join(save_path, "generator_final.h5"))
    discriminator.save(os.path.join(save_path, "discriminator_final.h5"))
    print("Modelos finales guardados.")

    # Generar gráfico de las pérdidas
    # plot_losses(g_losses_history, d_losses_history)


def plot_losses(g_losses_history, d_losses_history):
    plt.figure(figsize=(10, 5))

    # Curva de pérdida del generador
    plt.plot(g_losses_history, label="Generador (G Loss)", color="blue")

    # Curva de pérdida del discriminador
    plt.plot(d_losses_history, label="Discriminador (D Loss)", color="red")

    plt.title("Curvas de pérdida - Generador vs Discriminador")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()

    # Guardar el gráfico
    plt.savefig("gan_loss_curves.png")

    # Mostrar el gráfico
    plt.show()
