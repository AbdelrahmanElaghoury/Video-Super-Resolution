from model import RRN
import tensorflow as tf
import tensorflow_addons as tfa
from Load_Train import TrainDataGenerator

# Enviroment variables
epochs = 5
batch_size = 16
weight_decay = 5e-4
gamma = 0.1 
step_size = 60
scale_factor = 4
n_f = 128
n_b = 5
GT_path = 'Vimeo90k_256/train/'
input_shape = (64, 64, 118)

def main():
    loss_fn = tf.keras.losses.mae
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=weight_decay)
    model = RRN(n_f, n_b, scale_factor)
    data_gen = TrainDataGenerator(GT_path, batch_size, scale_factor).dataset()
    model.fit(data_gen, epochs, loss_fn, optimizer)

if __name__ == "__main__":
    main()