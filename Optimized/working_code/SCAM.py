from keras.layers import *
import tensorflow as tf


class SpatialAttention(Layer):
    """
    Spatial Attention Mechanism for 2D inputs.
    """

    def __init__(self, bias=False, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.bias = bias
        # Define the convolution layer to apply spatial-wise attention
        self.conv = Conv2D(
            filters=32,
            kernel_size=7,
            strides=1,
            padding='same',
            use_bias=self.bias,
            kernel_initializer='he_normal'
        )

    def attention_layer(self, x):
        # Compute max and average pooling along the spatial (height, width) axis
        max_pool = Lambda(lambda z: tf.reduce_max(z, axis=[1, 2], keepdims=True))(x)
        avg_pool = Lambda(lambda z: tf.reduce_mean(z, axis=[1, 2], keepdims=True))(x)

        # Concatenate max and average pooling results along the channel axis
        concat = Concatenate(axis=3)([max_pool, avg_pool])  # axis=3 for channel axis in 2D

        # Apply convolution to the concatenated result
        output = self.conv(concat)

        # Apply sigmoid activation
        output = tf.sigmoid(output)

        # Apply spatial-wise attention
        output = output * x
        return output


class ChannelAttention(Layer):
    """Constructs a Channel Attention module for 2D inputs.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3, name='Channel Attention'):
        super(ChannelAttention, self).__init__(name=name)
        self.avg_pool = GlobalAveragePooling2D()
        self.conv = Conv2D(1, kernel_size=k_size, padding='same', use_bias=False)
        self.sigmoid = tf.keras.activations.sigmoid


    def Attention_layer(self, x, ratio=8):
        channel_axis = -1
        channel = x.shape[channel_axis]

        # Shared dense layers to reduce the number of channels
        shared_layer_one = Dense(channel // ratio,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')

        # Global average pooling for the input feature map
        avg_pool = GlobalAveragePooling2D()(x)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        avg_pool = shared_layer_one(avg_pool)
        avg_pool = shared_layer_two(avg_pool)

        # Global max pooling for the input feature map
        max_pool = GlobalMaxPooling2D()(x)
        max_pool = Reshape((1, 1, channel))(max_pool)
        max_pool = shared_layer_one(max_pool)
        max_pool = shared_layer_two(max_pool)

        # Adding the results of average pooling and max pooling
        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)

        # Applying the attention map to the input
        return multiply([x, cbam_feature])





class SpatialAndChannelJointAttention(Layer):
    """
    Spatial and Channel Joint Attention Mechanism.
    This combines both Spatial and Channel Attention by averaging their outputs.
    """

    def __init__(self, bias=False, k_size=3, **kwargs):
        super(SpatialAndChannelJointAttention, self).__init__(**kwargs)
        self.bias = bias
        self.k_size = k_size

        # Define the convolution layer for spatial attention (same as in SpatialAttention)
        self.conv = Conv2D(
            filters=32,
            kernel_size=7,
            strides=1,
            padding='same',
            use_bias=self.bias,
            kernel_initializer='he_normal'
        )

    def Attention_layer(self, x):
        # Apply both spatial and channel attention
        spatial_attention = SpatialAttention()(x)
        channel_attention = ChannelAttention()(x)

        # Average the spatial and channel attention maps
        joint_attention_map = Add()([spatial_attention, channel_attention])

        return joint_attention_map



