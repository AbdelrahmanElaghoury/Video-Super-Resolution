import numpy as np
from PIL import Image
import tensorflow as tf
import scipy.ndimage.filters as fi

def load_sequence(sequence_path):
    GT = [Image.open(sequence_path + str(img) + '.png') for img in range(7)]
    return GT

def gkern(kernlen=13, nsig=1.6):
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)

def Gaussian_down_sample(sequence, scale):
    """
        sequence sahpe : [F, H, W, C]
    """
    assert scale in[2, 4], "Enter valid scale value [2, 4]"
    
    nsigma = scale * 0.4
    kernal_in = gkern(nsig=nsigma)
    gauss_kernel_2d = tf.convert_to_tensor(kernal_in, dtype=tf.float32)
    gauss_kernel = tf.tile(gauss_kernel_2d[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 3]) # 13*13*3*3
    
    pad_w, pad_h = 6, 6   # Filter padding
    padded_sequence = tf.pad(sequence, [[0,0], [pad_w,pad_w], [pad_h,pad_h], [0,0]], mode='REFLECT')
    
    # Pointwise filter that does nothing
    pointwise_filter = tf.eye(3, batch_shape=[1, 1])
    LR_seq = tf.nn.separable_conv2d(padded_sequence, gauss_kernel, pointwise_filter,
                                    strides=[1, scale, scale, 1], padding=[[0,0],[0,0],[0,0],[0,0]])
    return LR_seq

def data_preprocessing(GT_sequence_path, scale):
    GT = load_sequence(GT_sequence_path)
    GT = [np.asarray(img) for img in GT]
    GT = tf.cast(tf.convert_to_tensor(GT), tf.float32)
    LR = Gaussian_down_sample(GT, scale)
    LR = tf.concat([LR[1:2, :, :, :], LR], axis=0)
    return GT, LR