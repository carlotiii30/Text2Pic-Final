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
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    images_info = coco.loadImgs(image_ids)

    captions = []
    images = []

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    for img_info in images_info:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = f"{data_dir}/{file_name}"

        image = preprocess_image(image_path)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if "caption" in ann:
                captions.append(ann["caption"])
                images.append(image)

    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    images = np.array(images)
    captions = np.array(padded_sequences)

    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = (
        dataset.shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return dataset


def load_coco_subset(
    data_dir,
    annotation_file,
    batch_size=64,
    num_samples=500,
    tokenizer_path="data/tokenizer.pkl",
):
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()

    image_ids = image_ids[:num_samples]
    images_info = coco.loadImgs(image_ids)

    captions = []
    images = []

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    for img_info in images_info:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = f"{data_dir}/{file_name}"

        image = preprocess_image(image_path)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if "caption" in ann:
                captions.append(ann["caption"])
                images.append(image)

    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    images = np.array(images)
    captions = np.array(padded_sequences)

    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = (
        dataset.shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return dataset
