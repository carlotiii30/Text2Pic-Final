import os
import tensorflow as tf
import json


def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        annotations = json.load(f)

    image_id_to_caption = {}
    for annotation in annotations["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]
        if image_id in image_id_to_caption:
            image_id_to_caption[image_id].append(caption)
        else:
            image_id_to_caption[image_id] = [caption]

    return image_id_to_caption


def load_image_caption_pairs(
    image_dir, image_id_to_caption, subset_size=None, batch_size=16, img_size=(32, 32)
):
    image_paths = []
    captions = []

    for image_file in os.listdir(image_dir):
        image_id = int(
            image_file.split(".")[0]
        )  # Asumiendo que el nombre de la imagen es su ID
        if image_id in image_id_to_caption:
            image_path = os.path.join(image_dir, image_file)
            for caption in image_id_to_caption[image_id]:
                image_paths.append(image_path)
                captions.append(caption)

        if subset_size and len(image_paths) >= subset_size:
            break

    if subset_size:
        image_paths = image_paths[:subset_size]
        captions = captions[:subset_size]

    return image_paths, captions


def load_coco_dataset(
    image_paths,
    captions,
    batch_size=16,
    img_size=(32, 32),
    vocab_size=10000,
    max_length=60,
):
    def process_image(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = image / 255.0
        return image

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size, oov_token="<OOV>"
    )
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding="post"
    )

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    image_dataset = image_dataset.map(
        lambda x: process_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    caption_dataset = tf.data.Dataset.from_tensor_slices(padded_sequences)

    dataset = tf.data.Dataset.zip((image_dataset, caption_dataset))
    dataset = (
        dataset.shuffle(buffer_size=len(image_paths))
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    return dataset, tokenizer
