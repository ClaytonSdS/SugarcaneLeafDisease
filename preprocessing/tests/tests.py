# Checks that the libraries have been imported successfully

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

print("All the libraries are installed correctly! ✅")

# Check if TensorFlow is installed and its version
print("TensorFlow version:", tf.__version__)

# Creating a simple neural network test model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Display the model structure
model.summary()

print("Test model successfully created! ✅")