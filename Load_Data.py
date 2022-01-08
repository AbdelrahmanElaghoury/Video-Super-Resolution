import os
import random
import tensorflow as tf 
from Gaussian_DownSample import data_preprocessing

class TrainValidationDataGenerator:
    def __init__(self, dataset_path, batch_size, data_size, scale):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.data_size = data_size
        self.scale = scale

    def data_generator(self):
        sequences = os.listdir(self.dataset_path)
        random.shuffle(sequences)
        sequences = sequences[:self.data_size]

        for seq in sequences:
            sequences_path = self.dataset_path + str('%05d'%int(seq)) + '/'
            yield data_preprocessing(sequences_path, self.scale)
    
    def dataset(self):
        data_set = tf.data.Dataset.from_generator(self.data_generator,
                                                  output_signature=(tf.TensorSpec(shape=(7, 256, 256, 3), dtype=tf.float32),
                                                                    tf.TensorSpec(shape=(8, 64, 64, 3), dtype=tf.float32)))
        return data_set.batch(self.batch_size, drop_remainder=True)