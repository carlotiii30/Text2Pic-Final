import numpy as np


def train(generator, discriminator, combined, dataset, epochs=50, batch_size=64):
    for epoch in range(epochs):
        for batch in dataset:
            imgs, captions = batch

            if imgs.shape[0] != batch_size:
                continue

            noise = np.random.normal(0, 1, (batch_size, 80))
            combined_input = np.hstack((noise, captions))
            gen_imgs = generator.predict(combined_input)

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = combined.train_on_batch(combined_input, valid)

        print(f"Epoch {epoch}, FINALIZADA")
