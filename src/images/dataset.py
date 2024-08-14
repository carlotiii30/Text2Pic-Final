import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pycocotools.coco import COCO
import numpy as np
import pickle

from src.images.text_process import max_length


def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [64, 64])
    image = (image / 127.5) - 1
    return image


def preprocess_text(text, tokenizer, max_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    return padded_sequence


def load_coco_dataset(
    data_dir, annotation_file, batch_size=64, tokenizer_path="data/tokenizer.pkl"
):
    print("Iniciando la carga del dataset COCO...")
    
    coco = COCO(annotation_file)
    print("Archivo de anotaciones cargado.")
    
    image_ids = coco.getImgIds()
    images_info = coco.loadImgs(image_ids)

    captions = []
    images = []

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
    print("Tokenizer cargado.")

    for img_info in images_info:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = f"{data_dir}/{file_name}"

        image = preprocess_image(image_path)
        print(f"Imagen procesada: {file_name}")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if "caption" in ann:
                captions.append(ann["caption"])
                images.append(image)

    print("Todas las imágenes y captions han sido procesadas.")

    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    print("Textos convertidos a secuencias y secuencias rellenadas.")

    images = np.array(images)
    captions = np.array(padded_sequences)

    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = (
        dataset.shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    print("Dataset de TensorFlow creado.")

    print("Carga del dataset COCO completada.")
    return dataset


def load_coco_subset(
    data_dir,
    annotation_file,
    batch_size=64,
    num_samples=50000,
    tokenizer_path="data/tokenizer.pkl",
):
    print("Iniciando la carga del subset del dataset COCO...")

    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()

    image_ids = image_ids[:num_samples]
    images_info = coco.loadImgs(image_ids)
    print(f"Archivo de anotaciones cargado. Procesando {num_samples} imágenes.")

    captions = []
    images = []

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)
    print("Tokenizer cargado.")

    total_images = len(images_info)
    for idx, img_info in enumerate(images_info, start=1):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = f"{data_dir}/{file_name}"

        image = preprocess_image(image_path)
        print(f"Imagen procesada: {file_name} ({idx}/{total_images})")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if "caption" in ann:
                captions.append(ann["caption"])
                images.append(image)

    print(f"Todas las imágenes y captions han sido procesadas. Total: {total_images}")

    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    print("Textos convertidos a secuencias y secuencias rellenadas.")

    images = np.array(images)
    captions = np.array(padded_sequences)

    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = (
        dataset.shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    print("Dataset de TensorFlow creado.")

    print("Carga del subset del dataset COCO completada.")
    return dataset
