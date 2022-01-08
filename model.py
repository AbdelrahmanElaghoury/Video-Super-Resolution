import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from Architecture import hidden
import matplotlib.pyplot as plt

# Model class
class RRN(tf.keras.Model):
    def __init__(self, n_f, n_b, scale):
        super(RRN, self).__init__()
        self.hidden = hidden(n_f, n_b, scale)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.n_f = n_f
        
    def call(self, x, x_h, x_o, init):
        f1 = x[:,0,:,:,:]
        f2 = x[:,1,:,:,:]
        h,w = f1.shape[1:3]
        x_input = tf.concat([f1, f2], axis=-1)
        if init:
            x_h, x_o = self.hidden(x_input, x_h, x_o)
        else:
            x_o = self.down(x_o)
            x_h, x_o = self.hidden(x_input, x_h, x_o)
        
        x_o = tf.image.resize(f2, (h*self.scale, w*self.scale)) + tf.nn.depth_to_space(x_o, self.scale)
        return x_h, x_o
    
    def train_step(self, loss_fn, optimizer, LR, GT):
        B,F,_,_,_ = LR.shape
        output = []
        
        with tf.GradientTape() as tape:
            for frame_index in range(F-1):
                if not bool(frame_index):
                    # Initialize frame[-1], hidden_state[-1] and prediction[-1]
                    init_frame = tf.zeros_like(LR[:,0,:,:,:])
                    prediction = tf.repeat(init_frame, repeats= self.scale**2, axis=3)
                    hidden_state = tf.repeat(init_frame[:,:,:,:1], repeats= self.n_f, axis=-1)

                hidden_state, prediction = self(LR[:,frame_index:frame_index+2,:,:,:], hidden_state, prediction, not bool(frame_index))
                output.append(prediction)
                
            output = tf.stack(output, axis=1)
            loss = tf.reduce_mean(loss_fn(GT, output))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
        return output, loss
    
    def test_step(self, LR, GT=None, loss_fn=None):
        B,F,_,_,_ = LR.shape
        output = []
        for frame_index in range(F-1):
            if not bool(frame_index):
                # Initialize frame[-1], hidden_state[-1] and prediction[-1]
                init_frame = tf.zeros_like(LR[:,0,:,:,:])
                prediction = tf.repeat(init_frame, repeats= self.scale**2, axis=3)
                hidden_state = tf.repeat(init_frame[:,:,:,:1], repeats= self.n_f, axis=-1)
            
            hidden_state, prediction = self(LR[:,frame_index:frame_index+2,:,:,:], hidden_state, prediction, not bool(frame_index))
            output.append(prediction)

        output = tf.stack(output, axis=1)
        if GT != None:
            loss = tf.reduce_mean(loss_fn(GT, output))
            return output, loss
        return output

    def plot_GT_pred(self, GT, pred):
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(GT[0,0])
        axs[0].title.set_text('Ground Truth')
        axs[1].imshow(pred[0,0])
        axs[1].title.set_text('Network output')
        plt.show()
        
    def fit(self, train_data_generator, validation_data_generator, epochs, loss_fn, optimizer, check_point_path = ''):
        if check_point_path:
            self.load_weights(check_point_path)
    
        for epoch in range(epochs):
            t0 = time.time()
            train_epoch_loss = []
            validation_epoch_loss = []

            for batch_index, data in enumerate(train_data_generator):
                GT, LR = data
                prediction, batch_loss = self.train_step(loss_fn, optimizer, LR, GT)
                train_epoch_loss.append(batch_loss)

                if batch_index%20 == 0:
                    self.save_weights('check_point/model_weights')
                
            for data in validation_data_generator:
                GT, LR = data
                prediction, batch_loss = self.test_step(LR, GT, loss_fn)
                validation_epoch_loss.append(batch_loss)

            t1 = time.time()
            print("Epoch[{}/{}] Train_Loss: {:.4f} || Validation_Loss: {:.4f} || Timer: {:.4f} sec.".
            format(epoch+1, epochs, tf.reduce_mean(train_epoch_loss), tf.reduce_mean(validation_epoch_loss), (t1-t0)))

            # print(epoch_loss)
            self.plot_GT_pred(GT, prediction)
            train_loss = pd.DataFrame(train_epoch_loss)
            validation_loss = pd.DataFrame(validation_epoch_loss)


            train_loss.to_csv('train_VGG_Loss_' + str(epoch) + '.csv')
            validation_loss.to_csv('validation_VGG_Loss_' + str(epoch) + '.csv')


    def predict(self, LR_frames_path, check_point_path):
        self.load_weights(check_point_path)
        LR = [Image.open(LR_frames_path + '//' + image for image in os.listdir(LR_frames_path))]
        LR = [np.asarray(img) for img in LR]
        LR = tf.cast(tf.convert_to_tensor(LR), tf.float32)
        LR = tf.concat([LR[1:2, :, :, :], LR], axis=0)
        LR = tf.expand_dims(LR, axis=0)
        return  self.test_step(LR/255)

# Downsmaple the input
class PixelUnShuffle(tf.keras.Model):
    def __init__(self, scale):
        super(PixelUnShuffle, self).__init__()
        self.scale_factor = scale
    
    def call(self, x_o):
        x_o = tf.nn.space_to_depth(x_o, self.scale_factor)
        return x_o