import tensorflow as tf
import numpy as np

INPUT_FEATURE="img"


def cnn_model_fn(features=None, labels=None, mode=None, params=None):
    """ Three convolutional layer followed by max pooling
        In the finaly layer the output is flattened and put in to a 
        1024 neuron dense layer."""
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
    dense = tf.layers.dense(inputs=dense_input , units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=10)
    # Use a head to make it easier :)
    my_head = tf.contrib.estimator.multi_class_head(n_classes=10)
    return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=tf.train.AdamOptimizer(),
        logits=logits)

def main():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist") #Loads the data set
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    run_config = tf.estimator.RunConfig(save_checkpoints_secs = 120, 
	                                    keep_checkpoint_max = 3)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, 
        model_dir="model_dir",
        config=run_config
        )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"img": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
        )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"img": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
        )

    tf.logging.set_verbosity(tf.logging.INFO)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=300)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,start_delay_secs=30, throttle_secs=40)
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)



    def serving_input_fn():
        """Defines the features to be passed to the model during inference.
        Expects numpy array :)
        Returns:
            A tf.estimator.export.ServingInputReceiver
        """
        # Input to the serving function
        reciever_tensors = {INPUT_FEATURE: tf.placeholder(tf.float32, [None, 784])}
        # Convert give inputs to adjust to the model.
        features = {
        # Resize given images.
            INPUT_FEATURE: reciever_tensors[INPUT_FEATURE],
            }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                    features=features)



    mnist_classifier.export_savedmodel("export_dir", serving_input_fn)



if __name__=="__main__":
    main()