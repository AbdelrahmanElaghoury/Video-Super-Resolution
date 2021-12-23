import os
import numpy as np
from PIL import Image
import tensorflow as tf 
from Gaussian_DownSample import data_preprocessing

class TrainDataGenerator:
    def __init__(self, train_dataset_path , batch_size, scale):
        self.train_dataset_path = train_dataset_path
        self.batch_size = batch_size
        self.scale = scale

    def train_data_gen(self):
        sequences = os.listdir(self.train_dataset_path)
        for seq in sequences:
            sequences_path = self.train_dataset_path + str('%05d'%int(seq)) + '/'
            yield data_preprocessing(sequences_path, self.scale)
    
    def dataset(self):
        data_set = tf.data.Dataset.from_generator(self.train_data_gen,
                                                  output_signature=(tf.TensorSpec(shape=(7, 256, 256, 3), dtype=tf.float32),
                                                                    tf.TensorSpec(shape=(8, 64, 64, 3), dtype=tf.float32)))
        return data_set.batch(self.batch_size)