import os

def ensure_dir_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_model(dataset, model):
    model.fit(dataset, epochs=50)


def save_model_weights(model, filename):
    ensure_dir_exists(filename)
    if os.path.exists(filename):
        os.remove(filename)
    model.save_weights(filename)


def load_model_with_weights(model, filename):
    model.load_weights(filename)
    return model
