import tensorflow as tf

import model 
from utils import train_fn, eval_fn,serving_input_fn


def model_setup():
    train_input_fn = train_fn()
    eval_input_fn = eval_fn()
    return train_input_fn,eval_input_fn


def main ():
    """The main traning function for running this"""
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = 120, 
	                                    keep_checkpoint_max = 3)
    train_input_fn,eval_input_fn = model_setup()
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model.cnn_model_fn, 
        model_dir="model_dir",
        config=run_config
        )
    tf.logging.set_verbosity(tf.logging.INFO)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=100)
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
        start_delay_secs=30, 
        throttle_secs=40,
        exporters = exporter
        )
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
    print("lets do something")

    
if __name__=="__main__":
    main()