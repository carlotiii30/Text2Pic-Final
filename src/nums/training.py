from src.nums import builders, dataset
from src import utils

dataset = dataset.load_dataset()
generator, discriminator = builders.build_models()
cond_gan = builders.build_conditional_gan(generator, discriminator)
utils.train_model(dataset, cond_gan)
utils.save_model_weights(cond_gan, "cgan_nums.weights.h5")
