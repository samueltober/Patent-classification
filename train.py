
import os

from absl import app
from absl import flags

import tensorflow as tf
from official.nlp import optimization

from models.baseline_bert import build_model
from datasets.imdb import IMDB

import tensorflow_cloud as tfc

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    if FLAGS.use_cloud:
        tfc.run(
            entry_point=None,
            distribution_strategy=None,
            requirements_txt='requirements.txt',
            chief_config=tfc.COMMON_MACHINE_CONFIGS[FLAGS.machine_config],
            worker_count=0,
            stream_logs=False,
            docker_config=tfc.DockerConfig(image=FLAGS.image,
                                           parent_image=FLAGS.cache_from, image_build_bucket=FLAGS.GCP_bucket_images))

    checkpoint_path = os.path.join(
        "gs://", FLAGS.GCP_bucket_training, FLAGS.model_path, "save_at_{epoch}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path),
    ]

    print("Number of epochs", FLAGS.num_epochs)

    classifier_model = build_model()

    imdb = IMDB()

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = tf.metrics.BinaryAccuracy()

    steps_per_epoch = tf.data.experimental.cardinality(imdb.train_ds).numpy()
    num_train_steps = steps_per_epoch * FLAGS.num_epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    classifier_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    print('Training model')

    history = classifier_model.fit(x=imdb.train_ds,
                                   validation_data=imdb.val_ds,
                                   callbacks=callbacks,
                                   epochs=FLAGS.num_epochs)

    if tfc.remote():
        SAVE_PATH = os.path.join(
            "gs://", FLAGS.GCP_bucket_training, FLAGS.model_path, 'final')
        classifier_model.save(SAVE_PATH)


if __name__ == '__main__':
    flags.DEFINE_boolean(
        'use_cloud', False, 'If True Google Cloud will be used to train the model else locally.')
    flags.DEFINE_enum('machine_config', 'K80_1X', [
                      'CPU', 'K80_1X', 'V100_1X'], 'The machine configuration')
    flags.DEFINE_string('image', 'gcr.io/patent-classifier-327714/tf_cloud_train:02',
                        'Docker image URI')
    flags.DEFINE_string('cache_from', 'gcr.io/patent-classifier-327714/tf_cloud_train:01',
                        'Docker image URI to be used as a cache when building the new Docker image.')
    flags.DEFINE_string('GCP_bucket_training', 'kth-iamip-training',
                        'GCS bucket name to be used for storing trained models and logs')
    flags.DEFINE_string('GCP_bucket_images', 'kth-iamip-images',
                        'GCS bucket name to be used for building a Docker image via [Google Cloud Build](https://cloud.google.com/cloud-build/).')
    flags.DEFINE_integer('num_epochs', 5, 'Number of epochs')
    flags.DEFINE_string('model_path', 'bert-small',
                        'The path to where models are saved')

    app.run(main)
