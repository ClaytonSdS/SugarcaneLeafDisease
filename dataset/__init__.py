# SugarcaneLeafDisease/dataset/__init__.py

#from SugarcaneLeafDisease.preprocessing.labeling.labeling_tools import ImageLabeling
from ..preprocessing.labeling.labeling_tools import ImageLabeling


class LabeledDataset:
    def __init__(self):
        data_load = ImageLabeling()
        data_load.load_data(tfrecord_file_name="tfrecord_dataset")
        data = data_load.parsed_dataset.take(2310)
        return data