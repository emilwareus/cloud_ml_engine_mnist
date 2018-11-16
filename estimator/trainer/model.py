import tensorflow as tf
import numpy as np

INPUT_FEATURE="img"

def cnn_model_fn(features=None, labels=None, mode=None, params=None):
    """ Three convolutional layer followed by max pooling
        In the finaly layer the output is flattened and put in to a 
        1024 neuron dense layer."""
        # HERE HERE HERE HERE HERE HERE
        # The params in here is it a dict or hparams? Need to fix this in order to not have problems 
        # HERE HERE HERE HERE HERE HERE
    input_layer = tf.reshape(features[INPUT_FEATURE],[-1,28,28,1])
    layer_1=tf.layers.conv2d(
        inputs = input_layer,
        filters = 20,
        kernel_size=[5,5],
        padding = "same",
        activation=tf.nn.relu
    )
    max_pool_1 = pool1 = tf.layers.max_pooling2d(
        inputs=layer_1, 
        pool_size=[2, 2], 
        strides=2)
    layer_2 = tf.layers.conv2d(
        inputs=max_pool_1,
        filters = 40,
        kernel_size=[4,4],
        padding = "same",
        activation=tf.nn.relu,
    )
    max_pool_2 = pool1 = tf.layers.max_pooling2d(
        inputs=layer_2, 
        pool_size=[2, 2], 
        strides=2)
    layer_3 = tf.layers.conv2d(
        inputs=max_pool_2,
        filters = 60,
        kernel_size = [3,3],
        padding = "same",
        activation=tf.nn.relu
    )
    dense_input = tf.layers.flatten(layer_3)
    if params.layers:
        for neurons in params.layers:
            dense = tf.layers.dense(inputs=dense_input , units=int(neurons), activation=tf.nn.relu)
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, 
                training=mode == tf.estimator.ModeKeys.TRAIN
                )
    else:
        dense = tf.layers.dense(inputs=dense_input , units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, 
            training=mode == tf.estimator.ModeKeys.TRAIN
            )
    logits = tf.layers.dense(inputs=dropout, units=10)
    # Use a head to make it easier :)
    my_head = tf.contrib.estimator.multi_class_head(n_classes=10)
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.train.AdamOptimizer(learning_rate=float(params.learning_rate)),
        logits=logits)