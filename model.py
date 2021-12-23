import time
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
    
    def train_step(self, GT, LR, loss_fn, optimizer):
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
            loss = tf.reduce_sum(loss_fn(GT, output))/(B*7)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
        return output, loss
    
    def plot_GT_pred(self, GT, pred):
        fig, axs = plt.subplots(1, 2, figsize=(15, 15))
        axs[0].imshow(tf.cast(GT[0,0], tf.uint8))
        axs[0].title.set_text('Ground Truth')
        axs[1].imshow(tf.cast(pred[0,0], tf.uint8))
        axs[1].title.set_text('Network output')
        plt.show()
        
    def fit(self, data_generator, epochs, loss_fn, optimizer, load_from_check_point = False):
        if load_from_check_point:
            self.load_weights('check_point/model_weights')
            
        for epoch in range(epochs):
            for batch_index, data in enumerate(data_generator):
                t0 = time.time()
                GT, LR = data
                prediction, batch_loss = self.train_step(GT, LR, loss_fn, optimizer)
                
                # self.plot_GT_pred(GT, prediction)
                if batch_index%20 == 0:
                    self.save_weights('check_point/model_weights')  
                    
                t1 = time.time()
                
                print("Epoch[{}/{}],({}): Batch_Loss: {:.4f} || Timer: {:.4f} sec.".
                format(epoch, epochs, (batch_index+1), batch_loss, (t1-t0)))
        
# Down smaple the input
class PixelUnShuffle(tf.keras.Model):
    def __init__(self, scale):
        super(PixelUnShuffle, self).__init__()
        self.scale_factor = scale
    
    def call(self, x_o):
        x_o = tf.nn.space_to_depth(x_o, self.scale_factor)
        return x_o