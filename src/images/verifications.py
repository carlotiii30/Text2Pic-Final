import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.images.dataset import preprocess_image

max_length = 20


def verify_image_preprocessing(data_dir, annotation_file, num_samples=5):
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    images_info = coco.loadImgs(image_ids[:num_samples])

    for img_info in images_info:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        image_path = f"{data_dir}/{file_name}"

        image = preprocess_image(image_path)

        plt.figure()
        plt.imshow(
            (image.numpy() * 127.5 + 127.5).astype(np.uint8)
        )  # Desnormalizar para visualizaciónlización
        plt.axis("off")
        plt.title(file_name)
        plt.show()


def verify_text_preprocessing(data_dir, annotation_file, tokenizer_path='data/tokenizer.pkl', num_samples=5):
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    captions = []

    for img_id in image_ids[:num_samples]:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if "caption" in ann:
                captions.append(ann["caption"])

    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    for original, sequence, padded in zip(captions, sequences, padded_sequences):
        print(f"Original: {original}")
        print(f"Sequence: {sequence}")
        print(f"Padded: {padded}")
        print()

data_dir = "data/coco"
annotation_file = f"{data_dir}/annotations/captions_train2017.json"
images_dir = f"{data_dir}/train2017"
verify_image_preprocessing(images_dir, annotation_file)
verify_text_preprocessing(data_dir, annotation_file)