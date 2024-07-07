import os


def train_model(dataset, cond_gan):
    cond_gan.fit(dataset, epochs=50)


def save_model_weights(cond_gan, filename):
    if os.path.exists(filename):
        os.remove(filename)
    cond_gan.save_weights(filename)


def load_model_with_weights(filename, cond_gan):
    cond_gan.load_weights(filename)
    return cond_gan
