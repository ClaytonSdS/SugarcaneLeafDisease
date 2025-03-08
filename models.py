import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
from .custom_layers import CBAM, ResidualBlock1, ResidualBlock2
import gdown
import numpy as np
import cv2

class Model:
    def __init__(self, filepath=None):
        self.filepath = filepath or self.get_model_path()
        self.model = self.load_model()

    def get_model_path(self):
        file_name = "model_18.keras"

        # If the file already exists, no need to download again
        if not os.path.exists(file_name):
            file_id = "1f-ecWRJj1O-J6p-36zr6_1Z69rW88m6P"
            url = f"https://drive.google.com/uc?id={file_id}"
            print("Downloading the model...")
            gdown.download(url, file_name, quiet=False)

        return file_name

    def load_model(self):
        return tf.keras.models.load_model(self.filepath, custom_objects={'CBAM': CBAM, 'ResidualBlock1': ResidualBlock1, 'ResidualBlock2': ResidualBlock2})

    def predict_batch(self, images):
        #input_shape = self.model.input_shape[1:]
        #print(f"Expected input shape: {input_shape}")
        self.predicted = self.model.predict(images)
        self.predicted = tf.keras.activations.softmax(self.predicted)
        self.predicted_classes = np.argmax(self.predicted, axis=1) 
        return self.predicted_classes

    def decode(self):
        labels_inversed = {0: "healthy", 1: "mosaic", 2: "redrot", 3: "rust", 4: "yellow"}
        self.predicted_decoded = [labels_inversed[label] for label in self.predicted_classes]
        print(self.predicted_decoded)


class Model_18(Model):
    def __init__(self, filepath=None):
        super().__init__(filepath)
