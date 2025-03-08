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
        self.AvgPool = layers.GlobalAveragePooling2D(keepdims=True)
        self.MaxPool = layers.GlobalMaxPooling2D(keepdims=True)
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
        F = inputs
        MLP = lambda x: self.SharedMLP(x)
        M_c = tf.sigmoid(MLP(self.AvgPool(F)) + MLP(self.MaxPool(F)))
        M_c = tf.reshape(M_c, (-1, 1, 1, tf.shape(inputs)[-1]))
        F_prime = M_c * F
        AvgPool = tf.reduce_mean(F_prime, axis=-1, keepdims=True)
        MaxPool = tf.reduce_max(F_prime, axis=-1, keepdims=True)
        concat = tf.concat([AvgPool, MaxPool], axis=-1)
        M_s = self.conv(concat)
        F_2prime = F_prime * M_s
        return F_2prime

@register_keras_serializable()
class ResidualBlock1(layers.Layer):
    def __init__(self, l2_reg=0.001, **kwargs):
        super(ResidualBlock1, self).__init__(**kwargs)
        self.R1_1x1 = layers.Conv2D(32, (1, 1), padding='same')
        self.R1_conv1 = layers.Conv2D(32, (5, 5), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.R1_cbam = CBAM(reduction=1, kernel_size=7)
        self.R1_conv2 = layers.Conv2D(32, (3, 3), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.R1_add = layers.Add()

    def call(self, inputs):
        r1_shortcut = inputs
        r1_shortcut = self.R1_1x1(r1_shortcut)
        x = self.R1_conv1(inputs)
        x = self.R1_cbam(x)
        x = self.R1_conv2(x)
        x = self.R1_add([x, r1_shortcut])
        return x

@register_keras_serializable()
class ResidualBlock2(layers.Layer):
    def __init__(self, kernels=64, kernel_size=3, cbam_reduction=4, **kwargs):
        super(ResidualBlock2, self).__init__(**kwargs)
        self.R2_1x1 = layers.Conv2D(kernels, (1, 1), padding='same')
        self.R2_conv1 = layers.Conv2D(kernels, (kernel_size, kernel_size), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.R2_cbam = CBAM(reduction=cbam_reduction, kernel_size=7)
        self.R2_conv2 = layers.Conv2D(kernels, (kernel_size, kernel_size), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.R2_maxpool = layers.MaxPool2D(pool_size=(2, 2))
        self.R2_add = layers.Add()

    def call(self, inputs):
        r2_shortcut = inputs
        r2_shortcut = self.R2_maxpool(r2_shortcut)
        r2_shortcut = self.R2_1x1(r2_shortcut)
        x = self.R2_conv1(inputs)
        x = self.R2_cbam(x)
        x = self.R2_conv2(x)
        x = self.R2_maxpool(x)
        x = self.R2_add([x, r2_shortcut])
        return x
