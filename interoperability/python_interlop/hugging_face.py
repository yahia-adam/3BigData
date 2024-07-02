from datasets import load_dataset
from PIL import Image
import os

# Nom du dataset, remplace par le nom réel
dataset_name = "Voxel51/CVPR_2024_Papers"
dataset = load_dataset(dataset_name)

# Crée un dossier pour sauvegarder les images
os.makedirs("images", exist_ok=True)

# Parcourir et sauvegarder les images
for idx, example in enumerate(dataset['train']):
    print(idx)
    image = example['image']  # La colonne peut avoir un nom différent
    image = Image.fromarray(image)  # Convertir en objet PIL si nécessaire
    image.save(f"images/image_{idx}.jpg")
