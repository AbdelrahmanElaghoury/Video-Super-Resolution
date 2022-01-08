import os
from model import RRN
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from Load_Data import TrainValidationDataGenerator
from tensorflow.keras.applications import VGG19, ResNet50

# Enviroment variables
epochs = 50
batch_size = 4
weight_decay = 5e-4
scale_factor = 4
n_f = 128
n_b = 5
GT_train_path = "C://Users//Administrator//Videos//Vimeo90k_256//train//"
GT_validation_path = "C://Users//Administrator//Videos//Vimeo90k_256//validation//"
checkpoint_path = "checkpoints//VGG//model_weights"
lr_frames_path = "C://Users//Administrator//Desktop//Test data//Vid4//vid4_H_L_test//foliage//lr_x4"
input_shape = (64, 64, 118)

def ssim_loss(y_true, y_pred, max_val=1):
    return -1 * tf.image.ssim(y_true, y_pred, max_val)

def psnr_loss(y_true, y_pred, max_val=1):
    return -1 * tf.image.psnr(y_true, y_pred, max_val)

def vgg_loss(y_true, y_pred):
    import numpy as np
    y_true_features = []
    y_pred_features = []
    vgg_model = VGG19(include_top=False, input_shape=(256, 256, 3))
    vgg_model.trainable = False
    for seq_index in range(2):
        y_true_features.append(vgg_model(y_true[seq_index]))
        y_pred_features.append(vgg_model(y_pred[seq_index]))
    return tf.keras.losses.mean_squared_error(tf.stack(y_true_features, axis=0), tf.stack(y_pred_features, axis=0))

def resNet50_loss(y_true, y_pred):
    import numpy as np
    y_true_features = []
    y_pred_features = []
    resNet_model = ResNet50(include_top=False, input_shape=(256, 256, 3))
    resNet_model.trainable = False
    for seq_index in range(2):
        y_true_features.append(resNet_model(y_true[seq_index]))
        y_pred_features.append(resNet_model(y_pred[seq_index]))
    return tf.keras.losses.mean_squared_error(tf.stack(y_true_features, axis=0), tf.stack(y_pred_features, axis=0))

def main():
    loss_fn = vgg_loss
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=weight_decay)
    model = RRN(n_f, n_b, scale_factor)
    train_data_gen = TrainValidationDataGenerator(GT_train_path, batch_size, 400, scale_factor).dataset()
    validation_data_gen = TrainValidationDataGenerator(GT_validation_path, batch_size, 40, scale_factor).dataset()
    model.fit(train_data_gen, validation_data_gen, epochs, loss_fn, optimizer, checkpoint_path)

if __name__ == "__main__":
    main()