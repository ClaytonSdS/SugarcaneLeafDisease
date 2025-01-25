from labeling_tools import ImageLabeling
import os

labels = {
    "healthy": 0,
    "mosaic": 1,
    "redrot": 2,
    "rust": 3,
    "yellow": 4
}

dataset_dir = os.path.join(os.path.dirname(__file__), "../segmentation")

# Recording the segmentation data as tfrecord file
labeling = ImageLabeling(image_dataset_dir=dataset_dir,
                         labels=labels)

# 1. Gravar o arquivo TFRecord original
labeling.record(tfrecord_file_name="tfrecord_dataset")
labeling.split_tfrecord("tfrecord_dataset", num_parts=4)  # Dividindo em 3 partes, por exemplo
