import os
import tensorflow as tf
from .custom_layers import CBAM, Xception
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def calculate_padding(size:int, goal:int):
  if size < goal:
      pad = goal - size
      if pad % 2 == 0: return pad // 2, pad // 2
      else: return pad // 2 + 1, pad // 2

  else:
      return 0, 0


class Model:
    def __init__(self, filepath: str = None, verbose: bool = True):
        """
        Initializes the Model class by loading the Keras models for batch-based and patch-based predictions.

        Args:
            filepath (str, optional): Path to the Keras model for batch-based prediction.
            patch (str, optional): Path to the Keras model for patch-based prediction.
            verbose (bool, optional): If True, enables logging output to the terminal. Default is True.
        """
        self.verbose = verbose
        self.filepath = filepath
        self.decoded = None
        self.predicted = None
        self.model = self.load_model(path=self.filepath,
                                     custom_objects={'CBAM': CBAM, 'Xception': Xception})

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
        _, self.height, self.width, _ = resized.shape # height and width to be used in rescale_by_padding()
        return resized

    def size_error_check(self):
        if self.height != self.target_size[0] or self.width != self.target_size[1]:
            self.array2predict = self.rescale_by_upscaling(array=self.array2predict)  
            self.rescale_by_padding()        
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
                    raise ValueError(f"[ERROR] Expected 3 channels, but received {num_channels}")
            else:
                raise ValueError(f"[ERROR] Expected shape (batch, height, width, channels), but received {shape}")
                
        self.batch_size, self.height, self.width, self.channels = self.array2predict.shape 

    def patches_check_validation(self):
        if self.patch_size > min(self.height, self.width):
            raise ValueError(f"[ERROR] The maximum patch size for this configuration is {min(self.height, self.width)}")
        elif self.patch_size <= 0:
            raise ValueError(f"[ERROR] The minimum patch size must be 1")

    # Check and validate the existence of predicted and decoded tensors before plotting
    def plot_check(self):
        # Check if the predicted tensor exists
        if self.predicted is None:
            raise ValueError("[ERROR] No predicted tensor found. Please call model.predict() first.")
        
        # Check if the decoded labels exist
        if self.decoded is None:
            self.decode()

    def plot_image_predicted(self, image_index: int = 0):
        """
        Plot the model predictions into a grid-like plot.

        This method visualizes the model's predictions by displaying patches of 
        images and their predicted class labels with corresponding colors.

        Args:
            image_index (int): The index of the image to plot. 
                                The value should be in the range [0, len(self.patches)],
                                where `self.patches` contains the patches from the model's predictions.

        Returns:
            None: This method does not return anything, it only displays the plot.

        Raises:
            ValueError: If the `self.predicted`  value is not set correctly.
        
        Example:
            >>> model = Model()
            >>> predictions = model.predict(images)
            >>> decoded_labels = model.decode(predictions)
            >>> model.plot_image_predicted(image_index=0)
        """
        # Define the colors associated with each class label
        class_colors = {
            'healthy': (0, 0.5, 0, 0.4),  
            'mosaic': (0.678, 0.847, 0.902, 0.4),  
            'yellow': (1, 1, 0, 0.6),  
            'rust': (0.6, 0, 0.8, 0.4), 
            'redrot': (1, 0, 0, 0.6)  
        }

        # Perform a check to ensure that predicted and decoded values are available
        self.plot_check()

        # Reshaping the patches into a grid of images (each image is a patch)
        patches_to_plot = tf.reshape(self.patches, [*self.patches.shape[:-1], self.patch_size, self.patch_size, 3])

        # Initialize the plot with subplots arranged in a grid
        fig, axes = plt.subplots(self.rows, self.cols, figsize=(self.cols * 2, self.rows * 2))

        # Loop through each grid cell to plot the corresponding patch
        for i in range(self.rows):
            for j in range(self.cols):
                ax = axes[i, j]

                # Convert the tensor to a numpy array and normalize the image (values between 0 and 1)
                img = self.patches[image_index, i, j].numpy().astype(float) / 255.0

                # Ensure that `self.decoded` is not None
                if self.decoded is None:
                    self.decode()  # Decode if needed

                # Get the class label and its corresponding color
                label = self.decoded[image_index, i, j]
                color = class_colors[label]  # Retrieve the color corresponding to the class label

                # Create a colored overlay (filter) for the image using the RGB part of the color
                overlay = np.full((self.patch_size, self.patch_size, 3), color[:3])  # Create a matrix filled with the RGB color
                image_with_overlay = img * (1 - color[3]) + overlay * color[3]  # Blend the image with the color overlay

                # Display the image with the overlay applied
                ax.imshow(image_with_overlay)
                ax.axis('off')  # Hide axes to make the images "stick together"

        # Adjust layout to remove unnecessary whitespace between subplots
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)

        # Create a legend to explain the color mapping for each class label
        legend_labels = ['Healthy', 'Mosaic', 'Yellow', 'Rust', 'Redrot']
        legend_colors = [class_colors['healthy'], class_colors['mosaic'], class_colors['yellow'], class_colors['rust'], class_colors['redrot']]

        # Generate legend items with color using Line2D
        legend_elements = [Line2D([0], [0], color=color[:3], lw=4) for color in legend_colors]

        # Add the legend to the right side of the plot
        fig.legend(legend_elements, legend_labels, loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5), title="Legend", title_fontsize=12)

        # Display the plot
        plt.show()


    def predict(self, images: np.array, patch_size: int = 64, use_patches: bool = True):
        images = np.array(images)
        self.use_patches = use_patches
        self.patch_size = patch_size
        self.decoded = None                     #  decode status set to None

        self.target_size = (255, 255)
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
            self.predicted = self.model.predict(self.array2predict)  
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
        self.decoded = np.vectorize(labels_inversed.get)(tf.argmax(pred, axis=-1)) if self.decoded is not None else self.decoded
        
        return self.decoded


class Model_20(Model):
    """
    A subclass of the `Model` class, specifically designed for loading and predicting with the model 18 architecture.

    Args:
        verbose (bool, optional): If True, enables verbose logging throughout the model's operations, providing detailed information 
                                  about each step. Default is True.

    Inherits:
        Model: A parent class responsible for handling the base operations of model loading, prediction, and image processing.

    Example:
        >>> model = Model_20(verbose=True)
        >>> predictions = model.predict(images)
    
    Attributes:
        verbose (bool): The verbosity level for logging information during model operations.
        model (tf.keras.Model): The Keras model for predictions.
    """

    def __init__(self, verbose: bool=True):
        dir_path = os.path.dirname(os.path.abspath(__file__)) 
        path = os.path.join(dir_path, "model_20.keras")  
        super().__init__(filepath=path, verbose=verbose)
