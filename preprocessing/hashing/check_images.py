import os
import hashlib
import pandas as pd
from PIL import Image

# Calculates a unique identifier (hash) for each image.

def calculate_image_hash(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return hashlib.md5(image_data).hexdigest()

# Maps all the images in a directory and identifies duplicate images.

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

# Defines the path of the image directory.
current_directory = os.getcwd()
dataset_path = os.path.join(current_directory, "preprocessing/resizing")

image_hashes, duplicates = map_images(dataset_path)

# Shows duplicate images.
print("Imagens duplicadas encontradas:")
for dup, original in duplicates:
    print(f"Duplicado: {dup} | Original: {original}")

# Saves the results as CSV.
pd.DataFrame(list(image_hashes.items()), columns=['Hash', 'FilePath']).to_csv('image_mapping.csv', index=False)
pd.DataFrame(duplicates, columns=['DuplicatedImage', 'OriginalImage']).to_csv('duplicates_mapping.csv', index=False)


