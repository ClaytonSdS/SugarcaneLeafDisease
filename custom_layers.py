from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers
import tensorflow as tf


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
        self.sigmoid = 0#
        self.sMLP_1 = layers.Dense(num_channels, activation="relu")
        self.sMLP_2 = layers.Dense(num_channels // self.reduction, activation="relu")
        self.sMLP_3 = layers.Dense(num_channels)

        self.conv = layers.Conv2D(1, kernel_size=self.kernel_size, padding="same", activation="sigmoid")

    def SharedMLP(self, F):
        x = layers.Flatten()(F)
        x = self.sMLP_1(x)
        x = self.sMLP_2(x)
        x = self.sMLP_3(x)
        return x

    def call(self, inputs):
        F = inputs  # Feature map original

        # Channel Attention Module
        MLP = lambda x: self.SharedMLP(x)

        M_c = tf.keras.activations.sigmoid(MLP(self.AvgPool(F)) + MLP(self.MaxPool(F)))
        M_c = tf.reshape(M_c, (-1, 1, 1, tf.shape(inputs)[-1]))

        F_prime = M_c * F  # element-wise multiplication,  i.e., F' = F ⊙ Mc

        # Spatial Attention Module
        AvgPool = tf.reduce_mean(F_prime, axis=-1, keepdims=True)
        MaxPool = tf.reduce_max(F_prime, axis=-1, keepdims=True)
        concat = tf.concat([AvgPool, MaxPool], axis=-1)
        M_s = self.conv(concat)

        F_2prime = F_prime * M_s  # element-wise multiplication between F'⊙ Ms, i.e., F" = F'⊙ Ms

        return F_2prime

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({
            'reduction': self.reduction,
            'kernel_size': self.kernel_size
        })
        return config

@register_keras_serializable()
class ResidualBlock1(layers.Layer):
    def __init__(self, kernels=64, kernel_size=3, cbam_reduction=4, **kwargs):
        super(ResidualBlock1, self).__init__(**kwargs)
        # ResBlock2
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.cbam_reduction = cbam_reduction

        self.R2_1x1 = layers.Conv2D(kernels, (1, 1), padding='same')

        self.R2_conv1 = layers.Conv2D(kernels, (kernel_size, kernel_size), padding='same', use_bias=False)
        self.R2_bn1 = layers.BatchNormalization()
        self.R2_act1 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.R2_cbam = CBAM(reduction=cbam_reduction, kernel_size=7)

        self.R2_conv2 = layers.Conv2D(kernels, (kernel_size, kernel_size), padding='same', use_bias=False)
        self.R2_bn2 = layers.BatchNormalization()
        self.R2_act2 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.R2_add = layers.Add()
        self.R2_bn3 = layers.BatchNormalization() 

    def call(self, inputs):
        # Atalho
        r2_shortcut = self.R2_1x1(inputs)

        # Caminho principal
        x = self.R2_conv1(inputs)
        x = self.R2_bn1(x)
        x = self.R2_act1(x)

        x = self.R2_cbam(x)

        x = self.R2_conv2(x)
        x = self.R2_bn2(x)
        x = self.R2_act2(x)

        x = self.R2_add([x, r2_shortcut])
        x = self.R2_bn3(x)  

        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "kernels": self.kernels,
            "kernel_size": self.kernel_size,
            "cbam_reduction": self.cbam_reduction,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        kernels = config.pop('kernels')
        kernel_size = config.pop('kernel_size')
        cbam_reduction = config.pop('cbam_reduction')
        return cls(kernels=kernels, kernel_size=kernel_size, cbam_reduction=cbam_reduction, **config)

@register_keras_serializable()
class ResidualBlock2(layers.Layer):
    def __init__(self, kernels=64, kernel_size=3, cbam_reduction=4, **kwargs):
        super(ResidualBlock2, self).__init__(**kwargs)
        # ResBlock2
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.cbam_reduction = cbam_reduction

        self.R2_1x1 = layers.Conv2D(kernels, (1, 1), padding='same')

        self.R2_conv1 = layers.Conv2D(kernels, (kernel_size, kernel_size), padding='same', use_bias=False)
        self.R2_bn1 = layers.BatchNormalization()
        self.R2_act1 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.R2_cbam = CBAM(reduction=cbam_reduction, kernel_size=7)

        self.R2_conv2 = layers.Conv2D(kernels, (kernel_size, kernel_size), padding='same', use_bias=False)
        self.R2_bn2 = layers.BatchNormalization()
        self.R2_act2 = tf.keras.layers.LeakyReLU(alpha=0.1)

        self.R2_maxpool = layers.MaxPool2D(pool_size=(2, 2))
        self.R2_add = layers.Add()
        self.R2_bn3 = layers.BatchNormalization()  

    def call(self, inputs):
        # Atalho
        r2_shortcut = self.R2_maxpool(inputs)
        r2_shortcut = self.R2_1x1(r2_shortcut)

        # Caminho principal
        x = self.R2_conv1(inputs)
        x = self.R2_bn1(x)
        x = self.R2_act1(x)

        x = self.R2_cbam(x)

        x = self.R2_conv2(x)
        x = self.R2_bn2(x)
        x = self.R2_act2(x)

        x = self.R2_maxpool(x)

        # Soma residual
        x = self.R2_add([x, r2_shortcut])
        x = self.R2_bn3(x) 

        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "kernels": self.kernels,
            "kernel_size": self.kernel_size,
            "cbam_reduction": self.cbam_reduction,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        kernels = config.pop('kernels')
        kernel_size = config.pop('kernel_size')
        cbam_reduction = config.pop('cbam_reduction')
        return cls(kernels=kernels, kernel_size=kernel_size, cbam_reduction=cbam_reduction, **config)
