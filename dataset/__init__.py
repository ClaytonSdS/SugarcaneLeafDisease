# SugarcaneLeafDisease/dataset/__init__.py

# Importação relativa
from ..preprocessing.labeling.labeling_tools import ImageLabeling


class LabeledDataset:
    def __init__(self):
        # Carregando os dados a partir dos arquivos TFRecord
        data_load = ImageLabeling()
        data_load.load_data([
            "tfrecord_dataset/tfrecord_dataset_part_1.tfrecord",
            "tfrecord_dataset/tfrecord_dataset_part_2.tfrecord",
            "tfrecord_dataset/tfrecord_dataset_part_3.tfrecord"
        ])

        # Atribuindo a parsed_dataset à instância da classe
        self.parsed_dataset = data_load.parsed_dataset

    def __iter__(self):
        # Permite iterar diretamente sobre o parsed_dataset
        return iter(self.parsed_dataset)
