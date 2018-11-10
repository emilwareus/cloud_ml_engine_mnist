import tensorflow as tf
import numpy as np


def cnn_model_fn(features, labels, mode, params):
    """ Three convolutional layer NN for Mnist classification """
    # input_shape = [-1,28,28,1] # Assume black and whit
    input_layer = features["numpy_img"]
    layer_1=tf.layers.conv2d(
        inputs = input_layer,
        filters = 20,
        kernel_size=[3,3],
        padding = "same",
        activation=tf.nn.relu
    )
    max_pool_1 = pool1 = tf.layers.max_pooling2d(
        inputs=layer_1, 
        pool_size=[2, 2], 
        strides=2)
    layer_2 = tf.layers.conv2d(
        inputs=layemax_pool_1r_1,
        filters = 40,
        kernel_size=[3,3],
        padding = "same",
        activation=tf.nn.relu,
    )
    max_pool_2 = pool1 = tf.layers.max_pooling2d(
        inputs=layer_1, 
        pool_size=[2, 2], 
        strides=2)
    layer_3 = tf.layers.conv2d(
        inputs=layemax_pool_1r_1,
        filters = 60,
        kernels = [3,3],
        padding = "same",
        activation=tf.nn.relu
    )
    dense_input = tf.layers.flatten(layer_3)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # need to contiune from here 