import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
from .custom_layers import CBAM, ResidualBlock1, ResidualBlock2
import gdown
import numpy as np
import cv2
from skimage.transform import resize
import jax
import jax.numpy as jnp


class Model:
    def __init__(self, filepath=None, patch=None):
        self.filepath = filepath

        self.model_patch = self.load_model(path = patch)
        self.model = self.load_model(path = self.filepath, 
                                     custom_objects={'CBAM': CBAM, 'ResidualBlock1': ResidualBlock1, 'ResidualBlock2': ResidualBlock2})

    def load_model(self, path:str, custom_objects:dict=None):
        return tf.keras.models.load_model(path, custom_objects)

    #############################################################################################################################################################################

    def rescale_by_upscaling(self, height_width:tuple, height_width_goal:tuple):
        height, width = height_width
        height_goal, width_goal = height_width_goal

        scale_factor = min(height_goal / height, width_goal / width)

        # Calculating the new width and height, based on the scale factor got previously
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Rezing the image with the new (new_width, new_height) shape
        resized_image = np.array([resize(img, (new_height, new_width), anti_aliasing=True) for img in self.array2predict])
        return resized_image

    def calculate_padding(self, size, goal):
        """Calculate padding for a single dimension (height or width)"""
        if size < goal:
            pad = goal - size
            # Return how much padding to apply on both sides (left and right, or top and bottom)
            if pad % 2 == 0:
                return pad // 2, pad // 2
            else:
                return pad // 2 + 1, pad // 2  # Slightly larger padding on the right or bottom side
        else:
            return 0, 0  # No padding required

    def rescale_by_padding(self, height_width:tuple, height_width_goal:tuple):
        """Resize the image by padding to the required size."""
        height, width = height_width
        height_goal, width_goal = height_width_goal

        pad_height_left, pad_height_right = self.calculate_padding(height, height_goal)
        pad_width_left, pad_width_right = self.calculate_padding(width, width_goal)

        if pad_height_left > 0 or pad_width_left > 0:
            # Apply padding if necessary
            new_height = height + pad_height_left + pad_height_right
            new_width = width + pad_width_left + pad_width_right

            # Inform the user about the new shape after padding
            print(f"Expected an image-size of at least ({height_goal}, {width_goal}), "
                f"so, your image was resized via zero-padding to shape ({new_height}, {new_width}). "
                f"Padding applied: height={pad_height_left+pad_height_right}, width={pad_width_left+pad_width_right}")
            
            return np.pad(self.array2predict,
                        ((0, 0),  # No padding for batch dimension
                        (pad_height_left, pad_height_right),  # Padding for height
                        (pad_width_left, pad_width_right),  # Padding for width
                        (0, 0)),  # No padding for channels
                        mode='constant', constant_values=0)
        else:
            return self.array2predict


    #############################################################################################################################################################################

    def size_error_check(self, predicting_in_patches=False):
        size2batches = 212
        size2patches = 64

        batch, height, width, channels = self.array2predict.shape  # input already in the format (batch, height, width, channels)


        # not predicting in patches
        if not predicting_in_patches:
            if height != size2batches or width != size2batches:
                self.array2predict = self.rescale_by_upscaling(height_width = (height, width), height_width_goal=(size2batches, size2batches))   # rescaling the images if necessary — either upscaling it or downscaling it.
                self.array2predict = self.rescale_by_padding(height_width = (height, width), height_width_goal=(size2batches, size2batches))     # adding padding if necessary
                
        # predicting in patches:
        else:
            if height < size2patches or width < size2patches:
                self.array2predict = self.rescale_by_upscaling(height_width = (height, width), height_width_goal=(size2batches, size2batches))   # upscaling the images if necessary.
                self.array2predict = self.rescale_by_padding(height_width = (height, width), height_width_goal=(size2batches, size2batches))     # adding padding if necessary


    def shape_mismatch_error_check(self, data):
        """
        Check if the input data shape matches the expected format.

        Parameters:
        - data (numpy array or list): The input image or batch of images.
        
        Raises:
        - ValueError: If the input shape is invalid or the number of channels is not 3.
        """
        self.array2predict = np.array(data)  # Convert to NumPy array if not already
        shape = self.array2predict.shape  # Get the shape of the data

        # Mismatch checking: the model expects input in the format (batch, height, width, channels)
        if len(shape) != 4:
            if len(shape) == 3:  # Case of a single image (height, width, channels)
                self.array2predict = np.expand_dims(self.array2predict, axis=0)  # Add batch dimension
                
                # Verify the number of channels
                num_channels = self.array2predict.shape[-1]  # Getting the number of channels from the new expanded image
                
                # Check if the channels are different than 3
                if num_channels != 3:
                    raise ValueError(f"Expected images with 3 channels, but got {num_channels} channel(s)")
            else:
                raise ValueError(f"Expected shape (batch, height, width, channels), but got {shape}")


    #############################################################################################################################################################################


    def extract_patches_jax(self, images, patch_size=64):
        batch, height, width, channels = images.shape

        # Calcula o número de patches na altura e na largura
        n_patches_h = height // patch_size
        n_patches_w = width // patch_size
        n_patches = n_patches_h * n_patches_w  # Número total de patches por imagem

        # Índices para a extração dinâmica dos patches
        i_idx = jnp.arange(n_patches_h) * patch_size    # i.e. [0,1,2,3] becomes [0, 64, 128, 256]
        j_idx = jnp.arange(n_patches_w) * patch_size    # i.e. [0,1,2,3] becomes [0, 64, 128, 256]

        # Cria uma grade de índices
        i_idx, j_idx = jnp.meshgrid(i_idx, j_idx, indexing='ij')  # Grid with all combinations of i and j indexes: (n_patches_h, n_patches_w)
        i_idx = i_idx.flatten()
        j_idx = j_idx.flatten()

        def extract_single_image(img):
            """extract patches of a single image using JAX dynamic_slice"""
            patches = jax.vmap(lambda i, j: jax.lax.dynamic_slice(img, (i, j, 0), (patch_size, patch_size, channels)))(i_idx, j_idx)
            return patches  # (n_patches, 64, 64, 3)

        # Aplica a extração a todas as imagens do batch
        patches_batch = jax.vmap(extract_single_image)(images)  # (batch, n_patches, 64, 64, 3)
        
        return patches_batch

    #############################################################################################################################################################################
    

    def predict_in_paches(self, images):
        self.shape_mismatch_error_check(images) #  Check if the shape of the input data is correct
        self.size_error_check()                 #  Checking if image-size have a proper minimum size

        patches = self.extract_patches_jax(jnp.array(self.array2predict))  # (batch, n_patches, 64, 64, 3)

        # Realiza a predição em todos os patches
        predicted_patches = jax.vmap(self.model_patch.predict)(patches)

        # Converte para classes
        self.predicted_classes = jnp.argmax(predicted_patches, axis=-1)

        self.last_predict_form = "in_patches"
        return self.predicted_classes


    def predict_in_batches(self, images):
        """
        Perform batch predictions on input images.

        Parameters:
        - images (numpy array or list): The input images, expected to be in one of the following formats:
          1. (batch, height, width, 3) → Directly usable in model prediction.
          2. (height, width, 3) → A single image; it will be expanded to (1, height, width, 3).

        Returns:
        - numpy array: Predicted class indices for each input image.
        
        Raises:
        - ValueError: If the input shape is invalid or the number of channels is not 3.
        """

        self.shape_mismatch_error_check(images) #  Check if the shape of the input data is correct
        self.size_error_check()                 #  Checking if image-size have a proper minimum size
        
        self.predicted = self.model.predict(self.array2predict)
        self.predicted_classes = np.argmax(self.predicted, axis=1) 

        self.last_predict_form = "in_batches"
        return self.predicted_classes
    
    #############################################################################################################################################################################

    def convert_predictions_to_labels(self):
        """
        Decode the model's predicted class indices into labels.

        This function maps the model's output class indices (0, 1, 2, 3, 4) to 
        corresponding  (e.g., 'healthy', 'mosaic', 'yellow', 'redrot', 'rust'), 
        and prints the decoded labels.

        Returns:
        - list: A list of predicted class labels in string format (e.g., ['healthy', 'mosaic', ...]).
        """
        labels_inversed = {0:'healthy', 1:'mosaic', 2:'yellow', 3:'redrot', 4:'rust'}
        self.predicted_decoded = [labels_inversed[label] for label in self.predicted_classes]
        return self.predicted_decoded



class Model_18(Model):
    def __init__(self):
        dir_path = os.path.dirname(os.path.abspath(__file__)) 
        path = os.path.join(dir_path, "model18.keras")  
        patch = os.path.join(dir_path, "model18_patches.keras")  
        super().__init__(filepath=path, patch=patch)
