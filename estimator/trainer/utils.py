import tensorflow as tf 
import numpy as np 

mnist = tf.contrib.learn.datasets.load_dataset("mnist") #Loads the data set
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

INPUT_FEATURE="img"
BATCH_SIZE=100
# Check this out https://cloud.google.com/blog/products/gcp/easy-distributed-training-with-tensorflow-using-tfestimatortrain-and-evaluate-on-cloud-ml-engine
# We should convert it to TFRecords to make it even better. 
def load_data():
    """Funcioon to load the data, yet to be implemented completely"""
    mnist = tf.contrib.learn.datasets.load_dataset("mnist") #Loads the data set
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return None


def train_fn():
    """This function will return a training input functio"""
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"img": train_data},
            y=train_labels,
            batch_size=BATCH_SIZE,
            num_epochs=None,
            shuffle=True
            )
    return train_input_fn


def eval_fn():
    """This function will return a eval input functio"""
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"img": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
        )
    return eval_input_fn


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