import tensorflow as tf
import time

class PatentDataSet:
    FILE_PATH = './data'
    def __init__(self, batch_size=32, val_split=0.2, test_split=0.1, seed=42):
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.__load_data(seed)
        
    def __load_data(self, seed: int) -> None:
        """ Creates tf data sets from data folder.

        Args:
            seed (int): seed for shuffling data.
        """
        raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            directory=self.FILE_PATH,
            batch_size=1,
            seed=seed)
        ds_size = raw_train_ds.cardinality().numpy()
        test_size = int(self.test_split*ds_size)
        val_size = int(self.val_split*ds_size)
        train_size = ds_size - test_size - val_size
            
        self.train_ds = raw_train_ds.take(train_size).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
        self.test_ds = raw_train_ds.skip(train_size)
        self.val_ds = self.test_ds.skip(test_size).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
        self.test_ds = self.test_ds.take(test_size).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).cache()
    
    
if __name__ == "__main__":
    start = time.time()
    data = PatentDataSet()
    
    print(f"Training batches: {data.train_ds.cardinality().numpy()}")
    print(f"Validation batches: {data.val_ds.cardinality().numpy()}")
    print(f"Test batches: {data.test_ds.cardinality().numpy()}")
    print(f"Loading dataset took: {time.time() - start} s")