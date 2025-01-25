# SugarcaneLeafDisease/dataset/__init__.py
from ..preprocessing.labeling.labeling_tools import ImageLabeling


class LabeledDataset:
    def __init__(self):
        data_load = ImageLabeling()
        data_load.load_data([
            "tfrecord_dataset/tfrecord_dataset_part_1.tfrecord",
            "tfrecord_dataset/tfrecord_dataset_part_2.tfrecord",
            "tfrecord_dataset/tfrecord_dataset_part_3.tfrecord",
            "tfrecord_dataset/tfrecord_dataset_part_4.tfrecord"
        ])

        self.parsed_dataset = data_load.parsed_dataset

    def __iter__(self):
        return iter(self.parsed_dataset)
