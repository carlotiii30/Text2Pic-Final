import os
import pickle
from pycocotools.coco import COCO
from tensorflow.keras.preprocessing.text import Tokenizer

# Definir el tamaño del vocabulario y la longitud máxima
vocab_size = 27549
max_length = 60

# Directorio y archivo de anotaciones de COCO
data_dir = "data/coco"
annotation_file = os.path.join(data_dir, "annotations/captions_train2017.json")

# Cargar las anotaciones de COCO
coco = COCO(annotation_file)
image_ids = coco.getImgIds()
captions = []

# Extraer todas las captions
for img_id in image_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        if "caption" in ann:
            captions.append(ann["caption"])

# Crear y entrenar el Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(captions)

# Guardar el Tokenizer
with open("data/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

print(f"Total captions loaded: {len(captions)}")
