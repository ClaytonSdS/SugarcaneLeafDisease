import os
from ..preprocessing.labeling.labeling_tools import ImageLabeling

class LabeledDataset:
    def __init__(self):
        base_dir = os.path.dirname(__file__) 
        tfrecord_dir = os.path.join(base_dir, 'tfrecord_dataset') 
        
        tfrecord_files = [
            os.path.join(tfrecord_dir, "tfrecord_dataset_part_1.tfrecord"),
            os.path.join(tfrecord_dir, "tfrecord_dataset_part_2.tfrecord"),
            os.path.join(tfrecord_dir, "tfrecord_dataset_part_3.tfrecord"),
            os.path.join(tfrecord_dir, "tfrecord_dataset_part_4.tfrecord")
        ]

        data_load = ImageLabeling()
        data_load.load_data(tfrecord_files)

        self.parsed_dataset = data_load.parsed_dataset

    def __iter__(self):
        return iter(self.parsed_dataset)
