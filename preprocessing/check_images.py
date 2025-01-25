import os
import hashlib
import pandas as pd
from PIL import Image

# Calcula um identificador único (hash) para cada imagem.

def calculate_image_hash(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return hashlib.md5(image_data).hexdigest()

# Mapeia todas as imagens em um diretório e identifica imagens duplicadas.

def map_images(directory):
    image_hashes = {}
    duplicate_images = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                img_hash = calculate_image_hash(file_path)

                if img_hash in image_hashes:
                    duplicate_images.append((file_path, image_hashes[img_hash]))
                else:
                    image_hashes[img_hash] = file_path

    return image_hashes, duplicate_images

# Define o caminho do diretorio das imagens.
current_directory = os.getcwd()
dataset_path = os.path.join(current_directory, "dataset/preprocessing/resizing")

image_hashes, duplicates = map_images(dataset_path)

# Mostra imagens duplicadas
print("Imagens duplicadas encontradas:")
for dup, original in duplicates:
    print(f"Duplicado: {dup} | Original: {original}")

# Salva os resultados em CSV
pd.DataFrame(list(image_hashes.items()), columns=['Hash', 'FilePath']).to_csv('image_mapping.csv', index=False)
pd.DataFrame(duplicates, columns=['DuplicatedImage', 'OriginalImage']).to_csv('duplicates_mapping.csv', index=False)


