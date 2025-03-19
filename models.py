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
    def __init__(self, filepath: str = None, patch: str = None):
        """
        Initializes the Model class by loading the Keras models for batch-based and patch-based predictions.

        Args:
            filepath (str, optional): The file path to the Keras model for batch-based prediction. 
                                      If None, the model is not loaded. Default is None.
            patch (str, optional): The file path to the Keras model for patch-based prediction. 
                                  If None, the patch model is not loaded. Default is None.

        Initializes the `model_patch` and `model` attributes with the respective Keras models 
        if the corresponding file paths are provided. The models are loaded using the 
        `load_model` function, which also supports custom objects like `CBAM`, `ResidualBlock1`, and `ResidualBlock2`.

        Example:
            >>> model = Model(filepath="SugarcaneLeafDisease/model18.keras", patch="SugarcaneLeafDisease/model18_patches.keras")
            >>> print(model.model)        # Batch-based model
            >>> print(model.model_patch)  # Patch-based model
        """
        self.filepath = filepath
        self.model_patch = self.load_model(path=patch)
        self.model = self.load_model(path=self.filepath,
                                    custom_objects={'CBAM': CBAM, 'ResidualBlock1': ResidualBlock1, 'ResidualBlock2': ResidualBlock2})


    def load_model(self, path: str, custom_objects: dict = None):
        return tf.keras.models.load_model(path, custom_objects)


    def rescale_by_upscaling(self, array:np.array):
      # using patches
      if self.use_patches:
        array_flatten = tf.reshape(array, [self.batch_size*self.rows*self.cols, self.patch_size, self.patch_size, 3]) 
        print(f"[INFO] Reshaping patches (batch, rows, cols, pixels) → (b*r*c, patch_size, patch_size, channels): {array.shape} → {array_flatten.shape}")

        resized = tf.image.resize(array_flatten, self.target_size, preserve_aspect_ratio=True)        
        print(f"[INFO] Upscaling patches to match model input shape {self.target_size}: {array_flatten.shape} → {resized.shape}")

      # not using patches
      else:
        resized = tf.image.resize(array, self.target_size, preserve_aspect_ratio=True)        
      
      _, self.height, self.width, _ = resized.shape
      return resized

    def size_error_check(self):
        if self.height != self.target_size[0] or self.width != self.target_size[1]:
            self.array2predict = self.rescale_by_upscaling(array=self.array2predict)  


    def shape_mismatch_error_check(self, data:np.array):
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
        """
        Performs prediction on the input images using either a patch-based or batch-based approach.

        Args:
            images (np.ndarray): Input images with shape (batch, height, width, channels).
            patch_size (int, optional): The size of the patches to extract when using patch-based prediction. Default is 64.
            use_patches (bool, optional): If True, the images will be divided into patches before prediction. Default is True.
            use_patch_model (bool, optional): If True, uses the patch-based model instead of the full-image model. Default is False.

        Returns:
            tf.Tensor: Predicted output with shape (batch, rows, cols, 5) when using patches,  
                      where `rows` and `cols` define the grid layout of patches extracted from each image.
                      e.g., predict[0] shows the all the grids of image array in batch=0.  
                      In batch-based prediction, the output shape is (batch, 1, 1, 5), representing a single prediction per image.  
                      The last dimension corresponds to classification scores for 5 categories.
        """

        images = np.array(images)
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.use_patch_model = use_patch_model

        # Check to disable patch model if not using patches
        if not self.use_patches:
            self.use_patch_model = False
            print("[INFO] Not using patches, so use_patch_model set to False")

        self.target_size = ((212, 212) if not self.use_patch_model or not self.use_patches else (64, 64))  

        print(f'[INFO] Target size set to {self.target_size}')

        self.shape_mismatch_error_check(images)
        print("[INFO] Shape validation: completed successfully")

        self.patches_check_validation()
        print("[INFO] Patch validation: completed successfully")

        if self.use_patches:
            self.patches = tf.image.extract_patches(images=self.array2predict, sizes=[1, patch_size, patch_size, 1],
                                                    strides=[1, patch_size, patch_size, 1],
                                                    rates=[1, 1, 1, 1], padding="VALID")      
            self.rows, self.cols = self.patches.shape[1:3]                                    
            self.array2predict = self.patches

            self.height, self.width = (self.patch_size, self.patch_size)

            print(f"[INFO] Patch extraction completed - Output shape: {self.array2predict.shape}, Batch size: {self.batch_size}, Grid shape: ({self.rows},{self.cols}) = {self.rows*self.cols} patches, Patch size: ({self.patch_size},{self.patch_size}, {self.channels}) = {self.channels*self.patch_size**2} pixels")

        self.size_error_check()
        print("[INFO] Image size validation: completed successfully")

        if self.use_patches:  
            model_to_use = self.model_patch if self.use_patch_model else self.model   # selecting the proper model to predict in patches
            self.predicted = model_to_use.predict(self.array2predict)  
            self.predicted = tf.reshape(self.predicted, [self.batch_size, self.rows, self.cols, 5])  
            print(f"[INFO] Patch-based prediction completed - Output shape: {self.predicted.shape}")  

        else:  
            print("[INFO] Using batch-based model for prediction")  
            print(f"[INFO] Input shape set to {self.target_size}")  
            self.predicted = self.model.predict(self.array2predict)  
            self.predicted = tf.reshape(self.predicted, [self.batch_size, 1, 1, 5])  
            print(f"[INFO] Batch-based prediction completed - Output shape: {self.predicted.shape}")  

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

      return np.vectorize(labels_inversed.get)(
          tf.argmax(pred, axis=-1)
      )





class Model_18(Model):
    def __init__(self):
        dir_path = os.path.dirname(os.path.abspath(__file__)) 
        path = os.path.join(dir_path, "model18.keras")  
        patch = os.path.join(dir_path, "model18_patches.keras")  
        super().__init__(filepath=path, patch=patch)
