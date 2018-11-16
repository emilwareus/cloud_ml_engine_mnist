import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
import argparse

# Our own libaries and packages
import trainer.model as model 
from trainer.utils import train_fn, eval_fn,serving_input_fn


def model_setup():
    train_input_fn = train_fn()
    eval_input_fn = eval_fn()
    return train_input_fn,eval_input_fn
# gcloud ml-engine jobs submit training mnist_traning_31 --module-name trainer.task --region $REGION --package-path trainer --job-dir gs://cloud_ml_mnist --python-version 3.5 --runtime-version 1.10 --scale-tier STANDARD_1
# gcloud ml-engine jobs submit training mnist_traning_45 --module-name trainer.task --region $REGION --package-path trainer --job-dir gs://cloud_ml_mnist --python-version 3.5 --runtime-version 1.10 --scale-tier BASIC --config hptuning_config.yaml


def my_metric(labels, predictions):
    """We need to add a specific metric for the hyperparamter tunning job,
    the tag is important for the hyper_opt file"""
    # I know this will be returned based upon the head    
    logits = predictions['logits']
    return {'metric_hyp_opt': tf.metrics.accuracy(labels=labels, 
        predictions=tf.argmax(logits, 
        axis=1))
        }


def main(hparams):
    """The main traning function for running this"""
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = 120, 
	                                    keep_checkpoint_max = 3)
    train_input_fn,eval_input_fn = model_setup()
    mnist_classifier = tf.estimator.Estimator(
        model_fn=model.cnn_model_fn, 
        # params here should be dict or hparams? 
        # Need to make a choice about this
        # What is the best? 
        # Maybe i shouldn't sent the whole hparams just a part of it 
        params=hparams,
        model_dir=hparams.job_dir, # THIS FOLDER IS SUPER SUPER IMPORTANT TO HAVE CORRECT!
        config=run_config
        )
    # Need to add this for the hyperparamter tuning job to run
    mnist_classifier = tf.contrib.estimator.add_metrics(mnist_classifier, my_metric)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=int(hparams.max_steps))
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
        start_delay_secs=30, 
        throttle_secs=40,
        exporters = exporter
        )
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)


if __name__=="__main__":
    """Get the arguments and train the mdodel"""
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-filename',
        help='GCS file or local paths to data',
        nargs='+',
        default=None)
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        default="gs://cloud_ml_mnist"
        )
    parser.add_argument(
        '--max-steps',
        help='max number of traning steps',
        default=500
        )
    parser.add_argument(
        '--layers',
        help='neurons in each layer, pass nbr for each layer',
        nargs='+',
        default=[1024,512]
        )
    parser.add_argument(
        '--learning_rate',
        help='learning rate for the model',
        default=0.001
        )
    args, _ = parser.parse_known_args()
    hparams = hparam.HParams(**args.__dict__)
    main(hparams)
