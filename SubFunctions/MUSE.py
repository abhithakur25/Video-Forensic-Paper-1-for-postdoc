

import os
import numpy as np
import tensorflow as tf
import random as rn

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
from tensorflow.keras import backend as K
tf.random.set_seed(1234)




class TrainableAttention(tf.keras.layers.Layer):
    """
    Trainable attention layer.
    """
    def __init__(self, **kwargs):
        super(TrainableAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape)
        self._A = tf.Variable(np.identity(input_shape[2])*0.0, trainable=True, name='trainableattentionweights', dtype=tf.float32)
        super(TrainableAttention, self).build(input_shape)

    def call(self, x):
        re = K.dot(x, tf.keras.activations.sigmoid(self._A))
        return re


def excitation(input_feature, ratio=1, activation='elu', operation=''):
    """
    Excitation block for the excited attention model.
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    print('channel is', channel)
    initial = 'he_normal'
    se_feature = tf.keras.layers.Reshape((1, channel))(input_feature)
    se_feature = tf.keras.layers.Dense(int(channel * ratio), kernel_initializer=initial, use_bias=True, bias_initializer='zeros')(se_feature)
    se_feature = tf.keras.layers.Activation(activation)(se_feature)
    se_feature = tf.keras.layers.Dense(channel, activation='sigmoid', kernel_initializer=initial, use_bias=True, bias_initializer='zeros', name='excitation_' + str(round(ratio, 3)))(se_feature)
    if K.image_data_format() == 'channels_first':
        se_feature = tf.keras.layers.Permute((3, 1))(se_feature)
    return se_feature


def multi_excited_block(input_feature, features, num_excitations=None, activation='elu', operation='multiply', dropprob=0.05):
    """
    Multi-excited block for the excited attention model.
    """
    layers = []
    if num_excitations == None:
        num_excitations = max(2, int(np.sqrt(features)))
    ar = np.linspace(1./num_excitations, 1.0, num_excitations)
    print('linspaced', ar)
    for i in range(1, num_excitations+1):
        layers.append(excitation(input_feature, activation=activation, ratio=min(1, ar[i-1]), operation=operation))
    if len(layers) > 1:
        if operation in 'multiply':
            c = tf.keras.layers.Multiply()(layers)
        elif operation in 'add':
            c = tf.keras.layers.Add()(layers)
        elif operation in 'concatenate':
            c = tf.keras.layers.Concatenate(axis=1)(layers)
        elif operation in 'average':
            c = tf.keras.layers.Average()(layers)
    else:
        c = layers[0]
    c = tf.keras.layers.LayerNormalization(name='normalized_attention')(c)
    c = tf.keras.layers.multiply([input_feature, c], name='hadamard')
    c = tf.keras.layers.LayerNormalization(name='normalized_attention2')(c)
    c = TrainableAttention(name='trainableatt')(c)
    c = tf.keras.layers.Dense(features, activation=activation, use_bias=False)(c)
    c = tf.keras.layers.Dense(features, activation=activation, use_bias=False)(c)
    return c
