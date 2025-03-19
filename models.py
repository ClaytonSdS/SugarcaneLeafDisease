import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
from .custom_layers import CBAM, ResidualBlock1, ResidualBlock2
import numpy as np
import cv2
from skimage.transform import resize
import jax
import jax.numpy as jnp


def calculate_padding(size:int, goal:int):
  if size < goal:
      pad = goal - size
      if pad % 2 == 0: return pad // 2, pad // 2
      else: return pad // 2 + 1, pad // 2

  else:
      return 0, 0



class Model:
    def __init__(self, filepath: str = None, patch: str = None, verbose: bool = True):
        """
        Initializes the Model class by loading the Keras models for batch-based and patch-based predictions.

        Args:
            filepath (str, optional): Path to the Keras model for batch-based prediction.
            patch (str, optional): Path to the Keras model for patch-based prediction.
            verbose (bool, optional): If True, enables logging output to the terminal. Default is True.
        """
        self.verbose = verbose
        self.filepath = filepath
        self.model_patch = self.load_model(path=patch)
        self.model = self.load_model(path=self.filepath,
                                     custom_objects={'CBAM': CBAM, 'ResidualBlock1': ResidualBlock1, 'ResidualBlock2': ResidualBlock2})

    def print_verbose(self, message: str):
        if self.verbose:
            print(message)

    def load_model(self, path: str, custom_objects: dict = None):
        return tf.keras.models.load_model(path, custom_objects)

    def rescale_by_padding(self):
        pad_height_left, pad_height_right = calculate_padding(self.height, self.target_size[0])
        pad_width_left, pad_width_right = calculate_padding(self.width, self.target_size[1])

        # Apply padding if necessary
        if any(p > 0 for p in [pad_height_left, pad_height_right, pad_width_left, pad_width_right]):

            new_height = self.height + pad_height_left + pad_height_right
            new_width = self.width + pad_width_left + pad_width_right

            self.array2predict = np.pad(self.array2predict,
                          ((0, 0),                                # No padding for batch dimension
                           (pad_height_left, pad_height_right),   # Padding for height
                           (pad_width_left, pad_width_right),     # Padding for width
                           (0, 0)),                               # No padding for channels
                          mode='constant', constant_values=0)
            
            # Print message if verbose is enabled
            self.print_verbose(f"[INFO] Padding applied, new shape: {self.array2predict.shape}")

        else:
          self.print_verbose(f"[INFO] No need for padding. The current input shape is already {self.array2predict.shape}, which matches the target size {self.target_size}.")


    def rescale_by_upscaling(self, array: np.array):
        resized = tf.image.resize(array, self.target_size, preserve_aspect_ratio=True)    
        self.print_verbose(f"[INFO] Upscaling patches to match model input shape {self.target_size}: {array.shape} → {resized.shape}")
        _, self.height, self.width, _ = resized.shape
        return resized

    def size_error_check(self):
        if self.height != self.target_size[0] or self.width != self.target_size[1]:
            self.array2predict = self.rescale_by_upscaling(array=self.array2predict)  
            self.arra2predict = self.rescale_by_padding()        
        else:
            self.print_verbose(f"[INFO] No need for upscaling. The current input shape is already {self.array2predict.shape}, which matches the target size {self.target_size}.")

    def shape_mismatch_error_check(self, data: np.array):
        self.array2predict = np.array(data)  
        shape = self.array2predict.shape    
        if len(shape) != 4:
            if len(shape) == 3:  
                self.array2predict = np.expand_dims(self.array2predict, axis=0)  
                num_channels = self.array2predict.shape[-1]  
                if num_channels != 3:
                    raise ValueError(f"Expected 3 channels, but received {num_channels}")
            else:
                raise ValueError(f"Expected shape (batch, height, width, channels), but received {shape}")
                
        self.batch_size, self.height, self.width, self.channels = self.array2predict.shape 

    def patches_check_validation(self):
        if self.patch_size > min(self.height, self.width):
            raise ValueError(f"The maximum patch size for this configuration is {min(self.height, self.width)}")
        elif self.patch_size <= 0:
            raise ValueError(f"The minimum patch size must be 1")

    def predict(self, images: np.array, patch_size: int = 64, use_patches: bool = True, use_patch_model: bool = False):
        images = np.array(images)
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.use_patch_model = use_patch_model

        if not self.use_patches:
            self.use_patch_model = False
            self.print_verbose("[INFO] Not using patches, so use_patch_model set to False")

        self.target_size = ((212, 212) if not self.use_patch_model or not self.use_patches else (64, 64))  
        self.print_verbose(f'[INFO] Target size set to {self.target_size}')

        self.shape_mismatch_error_check(images)
        self.print_verbose("[INFO] Shape validation: completed successfully")

        self.patches_check_validation()
        self.print_verbose("[INFO] Patch validation: completed successfully")

        if self.use_patches:
            self.patches = tf.image.extract_patches(images=self.array2predict, sizes=[1, patch_size, patch_size, 1],
                                                    strides=[1, patch_size, patch_size, 1],
                                                    rates=[1, 1, 1, 1], padding="VALID")     
            self.rows, self.cols = self.patches.shape[1:3]                                    
            self.array2predict = self.patches
            self.print_verbose(f"[INFO] Patch extraction completed - Output shape: {self.array2predict.shape}, Batch size: {self.batch_size}, Grid shape: ({self.rows},{self.cols}) = {self.rows*self.cols} patches, Patch size: ({self.patch_size},{self.patch_size}, {self.channels}) = {self.channels*self.patch_size**2} pixels")

            self.array2predict = tf.reshape(self.array2predict, [self.batch_size*self.rows*self.cols, self.patch_size, self.patch_size, 3]) 
            self.print_verbose(f"[INFO] Reshaping patches (batch, rows, cols, pixels) → (b*r*c, patch_size, patch_size, channels): {self.patches.shape} → {self.array2predict.shape}")

            self.height, self.width = (self.patch_size, self.patch_size)

        self.size_error_check()
        self.print_verbose("[INFO] Image size validation: completed successfully")

        if self.use_patches:  
            model_to_use = self.model_patch if self.use_patch_model else self.model   
            self.predicted = model_to_use.predict(self.array2predict)  
            self.predicted = tf.reshape(self.predicted, [self.batch_size, self.rows, self.cols, 5])  
            self.print_verbose(f"[INFO] Patch-based prediction completed - Output shape: {self.predicted.shape}")  
        else:  
            self.print_verbose("[INFO] Using batch-based model for prediction")  
            self.print_verbose(f"[INFO] Input shape set to {self.target_size}")  
            self.predicted = self.model.predict(self.array2predict)  
            self.predicted = tf.reshape(self.predicted, [self.batch_size, 1, 1, 5])  
            self.print_verbose(f"[INFO] Batch-based prediction completed - Output shape: {self.predicted.shape}")  

        return self.predicted

    def decode(self, pred: tf.Tensor = None):
        """
        Decodes the model's prediction tensor into class labels.

        Args:
            pred (tf.Tensor, optional): A tensor containing the model's predictions. If None, 
                                        the function uses `self.predicted`. Default is None.

        Returns:
            np.ndarray: An array containing the predicted class labels as strings.

        Example:
            >>> model = Model()
            >>> predictions = model.predict(images)
            >>> decoded_labels = model.decode(predictions)
            >>> print(decoded_labels)
            [['healthy' 'mosaic' 'yellow']
            ['redrot' 'rust' 'healthy']]
        """
        labels_inversed = {0: 'healthy', 1: 'mosaic', 2: 'yellow', 3: 'redrot', 4: 'rust'}

        pred = pred if pred is not None else self.predicted

        return np.vectorize(labels_inversed.get)(tf.argmax(pred, axis=-1))


class Model_18(Model):
    """
    A subclass of the `Model` class, specifically designed for loading and predicting with the model 18 architecture.

    Args:
        verbose (bool, optional): If True, enables verbose logging throughout the model's operations, providing detailed information 
                                  about each step. Default is True.

    Inherits:
        Model: A parent class responsible for handling the base operations of model loading, prediction, and image processing.

    Example:
        >>> model = Model_18(verbose=True)
        >>> predictions = model.predict(images)
    
    Attributes:
        verbose (bool): The verbosity level for logging information during model operations.
        model (tf.keras.Model): The Keras model for batch-based predictions.
        model_patch (tf.keras.Model): The Keras model for patch-based predictions.
    """

    def __init__(self, verbose: bool=True):
        dir_path = os.path.dirname(os.path.abspath(__file__)) 
        path = os.path.join(dir_path, "model18.keras")  
        patch = os.path.join(dir_path, "model18_patches.keras")  
        super().__init__(filepath=path, patch=patch, verbose=verbose)
