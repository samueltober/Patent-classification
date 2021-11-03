import tensorflow as tf
import os
import shutil

class IMDB():

    def __init__(self):
        url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

        dataset = tf.keras.utils.get_file('aclImdb_v1.tar.gz', url,
                                          untar=True, cache_dir='../',
                                          cache_subdir='')

        dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

        train_dir = os.path.join(dataset_dir, 'train')

        # remove unused folders to make it easier to load the data
        remove_dir = os.path.join(train_dir, 'unsup')
        shutil.rmtree(remove_dir)

        AUTOTUNE = tf.data.AUTOTUNE
        batch_size = 32
        seed = 42

        raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            os.path.join(os.path.dirname(dataset), 'aclImdb/train'),
            batch_size=batch_size,
            validation_split=0.2,
            subset='training',
            seed=seed)

        self.class_names = raw_train_ds.class_names
        self.train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE).take(50)

        val_ds = tf.keras.preprocessing.text_dataset_from_directory(
            os.path.join(os.path.dirname(dataset), 'aclImdb/train'),
            batch_size=batch_size,
            validation_split=0.2,
            subset='validation',
            seed=seed)

        self.val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE).take(50)

        test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            os.path.join(os.path.dirname(dataset), 'aclImdb/test'),
            batch_size=batch_size)

        self.test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE).take(50)
