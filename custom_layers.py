from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers
import tensorflow as tf

@register_keras_serializable()
class XCeption(layers.Layer):
    def __init__(self, filters=16, kernel_size=3, **kwargs):
        super(XCeption, self).__init__(**kwargs)
        self.filters = filters  # Number of filters for 1x1 convolutions
        self.kernel_size = kernel_size  # Kernel size for depthwise convolution

    def build(self, input_shape):
        num_channels = input_shape[-1]  # Number of input channels (e.g., RGB = 3)

        # 1x1 convolution (Pointwise) to reduce the number of channels (it reduces the number of channels from num_channels to filters.)
        self.normalization = layers.Conv2D(self.filters, kernel_size=1, padding="same", use_bias=False) # Input shape: (batch_size, height, width, num_channels) Output shape: (batch_size, height, width, filters)

        # Depthwise convolution 3x3, applied to each channel separately
        self.depthwise = layers.DepthwiseConv2D(kernel_size=self.kernel_size, padding="same", use_bias=False) # Input shape: (batch_size, height, width, filters) Output shape: (batch_size, height, width, filters)

        # Pointwise convolution 1x1 to recombine information across channels
        self.pointwise = layers.Conv2D(self.filters, kernel_size=1, padding="same", use_bias=False) # Input shape: (batch_size, height, width, filters)  Output shape: (batch_size, height, width, filters)

        # Batch normalization layer to stabilize training
        self.bn = layers.BatchNormalization() # normalizes activations | Output shape: (batch_size, height, width, filters)

    def call(self, inputs):
        x = self.normalization(inputs)  # Apply 1x1 normalization (reduce channels)
        # Shape after normalization: (batch_size, height, width, filters)

        x = self.depthwise(x)  # Apply depthwise convolution (3x3)
        # Shape after depthwise convolution: (batch_size, height, width, filters)

        x = self.pointwise(x)  # Apply pointwise convolution (1x1)
        # Shape after pointwise convolution: (batch_size, height, width, filters)

        x = self.bn(x)  # Apply batch normalization
        # Shape after batch normalization: (batch_size, height, width, filters)

        return x

    def get_config(self):
        # Get the configuration of the layer, including its parameters
        base_config = super().get_config()
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # Recreate the layer from the configuration dictionary
        filters = config.pop('filters')
        kernel_size = config.pop('kernel_size')
        return cls(filters=filters, kernel_size=kernel_size, **config)

@register_keras_serializable()
class Xception(layers.Layer):
    def __init__(self, filters=16, kernel_size=3, **kwargs):
        super(Xception, self).__init__(**kwargs)
        self.filters = filters  # Number of filters for 1x1 convolutions
        self.kernel_size = kernel_size  # Kernel size for depthwise convolution

    def build(self, input_shape):
        num_channels = input_shape[-1]  # Number of input channels (e.g., RGB = 3)

        # 1x1 convolution (Pointwise) to reduce the number of channels (it reduces the number of channels from num_channels to filters.)
        self.normalization = layers.Conv2D(self.filters, kernel_size=1, padding="same", use_bias=False) # Input shape: (batch_size, height, width, num_channels) Output shape: (batch_size, height, width, filters)

        # Depthwise convolution 3x3, applied to each channel separately
        self.depthwise = layers.DepthwiseConv2D(kernel_size=self.kernel_size, padding="same", use_bias=False) # Input shape: (batch_size, height, width, filters) Output shape: (batch_size, height, width, filters)

        # Pointwise convolution 1x1 to recombine information across channels
        self.pointwise = layers.Conv2D(self.filters, kernel_size=1, padding="same", use_bias=False) # Input shape: (batch_size, height, width, filters)  Output shape: (batch_size, height, width, filters)

        # Batch normalization layer to stabilize training
        self.bn = layers.BatchNormalization() # normalizes activations | Output shape: (batch_size, height, width, filters)

    def call(self, inputs):
        x = self.normalization(inputs)  # Apply 1x1 normalization (reduce channels)
        # Shape after normalization: (batch_size, height, width, filters)

        x = self.depthwise(x)  # Apply depthwise convolution (3x3)
        # Shape after depthwise convolution: (batch_size, height, width, filters)

        x = self.pointwise(x)  # Apply pointwise convolution (1x1)
        # Shape after pointwise convolution: (batch_size, height, width, filters)

        x = self.bn(x)  # Apply batch normalization
        # Shape after batch normalization: (batch_size, height, width, filters)

        return x

    def get_config(self):
        # Get the configuration of the layer, including its parameters
        base_config = super().get_config()
        config = {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # Recreate the layer from the configuration dictionary
        filters = config.pop('filters')
        kernel_size = config.pop('kernel_size')
        return cls(filters=filters, kernel_size=kernel_size, **config)




@register_keras_serializable()
class CBAM(layers.Layer):
    def __init__(self, reduction=16, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.reduction = reduction
        self.kernel_size = kernel_size

    def build(self, input_shape):
        num_channels = input_shape[-1]

        self.AvgPool = layers.GlobalAveragePooling2D(keepdims=True)  # average-pooled features
        self.MaxPool = layers.GlobalMaxPooling2D(keepdims=True)  # max-pooled features

        # Shared MLP - built-in
        self.sMLP_1 = layers.Dense(num_channels, activation="relu")
        self.sMLP_2 = layers.Dense(num_channels // self.reduction, activation="relu")
        self.sMLP_3 = layers.Dense(num_channels)

        # 🔹 Atenção Espacial (Spatial Attention) - Será adicionada depois
        self.conv = layers.Conv2D(1, kernel_size=self.kernel_size, padding="same", activation="sigmoid")

    def SharedMLP(self, F):
        #x = layers.Flatten()(F)
        x = self.sMLP_1(F)
        x = self.sMLP_2(x)
        x = self.sMLP_3(x)
        return x

    def call(self, inputs):
        F = inputs  # Feature map original

        # Channel Attention Module
        MLP = lambda x: self.SharedMLP(x)
        M_c = tf.keras.activations.sigmoid(MLP(self.AvgPool(F)) + MLP(self.MaxPool(F)))
        #M_c = tf.reshape(M_c, (-1, 1, 1, tf.shape(inputs)[-1]))

        F_prime = M_c * F  # element-wise multiplication,  i.e., F' = F ⊙ Mc

        # Spatial Attention Module
        AvgPool = tf.reduce_mean(F_prime, axis=-1, keepdims=True)
        MaxPool = tf.reduce_max(F_prime, axis=-1, keepdims=True)
        concat = tf.concat([AvgPool, MaxPool], axis=-1)
        M_s = self.conv(concat)

        F_2prime = F_prime * M_s  # element-wise multiplication between F'⊙ Ms, i.e., F" = F'⊙ Ms

        return F_2prime

    def get_config(self):
        # Retorna os parâmetros necessários para reconstruir a camada
        config = super(CBAM, self).get_config()
        config.update({
            'reduction': self.reduction,
            'kernel_size': self.kernel_size
        })
        return config