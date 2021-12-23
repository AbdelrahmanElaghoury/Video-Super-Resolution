import functools
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

# Concatenate the Residual Blocks
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return tf.keras.Sequential(layers)

# Residual Blocks 
class Residual_Blocks(tf.keras.Model):
    '''Residual block w/o BN
    -+-Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, n_f):
        super(Residual_Blocks, self).__init__()
        self.conv1 = Conv2D(filters=n_f, kernel_size=(3,3), padding='same', activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv2 = Conv2D(filters=n_f, kernel_size=(3,3), padding='same', kernel_initializer=tf.keras.initializers.HeNormal())
        
    def call(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity

# Hidden layers of the network
class hidden(tf.keras.Model):
    def __init__(self, n_f, n_b, scale):
        super(hidden, self).__init__()
        self.conv1 = Conv2D(filters=n_f, kernel_size=(3,3), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.HeNormal())
        basic_block = functools.partial(Residual_Blocks, n_f=n_f)
        self.residual_blocks = make_layer(basic_block, n_b)
        self.conv_h = Conv2D(filters=n_f, kernel_size=(3,3), activation='relu', padding='same', name='hidden_state', kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv_o = Conv2D(filters=scale*scale*3, kernel_size=(3,3), padding='same', name='output', kernel_initializer=tf.keras.initializers.HeNormal())
        
    def call(self, X, h, o):
        x_input = tf.concat([X, tf.cast(h, tf.float32), tf.cast(o, tf.float32)], axis=-1)
        x = self.conv1(x_input)
        x = self.residual_blocks(x)
        x_h = self.conv_h(x)
        x_o = self.conv_o(x)
        return x_h, x_o