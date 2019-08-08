# import statements needed for the functions below
import tensorflow as tf
from tensorflow.contrib.layers import flatten


# LeNet-5 architecture implementation using TensorFlow
# x - placeholders for pictures
def le_net_5(x, standard_dev, receptive_field_1, filters_1, receptive_field_2, filters_2, activation):
    activations = ("relu", "relu6", "elu", "selu", "softplus", "softsign", "sigmoid", "tanh")
    # print(activations[activation])
    activation = getattr(tf.nn, activations[activation])  # it might not work - maybe change variable name?

    # Layer 1 : Convolutional Layer. Input = 32x32x1, Output = 28x28x1.
    # this means we accept pictures of 32x32 and will do 6 convolution
    # every convolution will go through all layers - in this case only one layer because this is greyscale picture
    # not sure why stddev is 0.1 maybe it yield better results with stddev=1
    # not sure why mean is at zero, I thought that weights should be positive...
    # they will be most probably nullified by relu but not sure to which extent
    # format is a 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # in channels is 1 because we have depth of 1 and out channels is 6 because this is what model lenet says
    # conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = 0.1))
    conv1_w = tf.Variable(tf.truncated_normal(shape=[receptive_field_1, receptive_field_1, 1, filters_1],



                                              mean=0, stddev=standard_dev))
    # print("conv1_w.shape variable: ", conv1_w.shape)

    # not sure why bias is not random
    conv1_b = tf.Variable(tf.zeros(filters_1))
    # conv1_b = tf.Variable(tf.zeros(8))
    # print("conv1_b.shape : zeroed", conv1_b.shape)

    # padding is Valid means that no padding needed, anyway we added padding before
    # maybe it would save memory to do padding as part of the graph calculation or
    # maybe this way it is faster cause it saves GPU cycles
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # print("conv1.shape conv2d :", conv1.shape)

    # Activation
    # not sure why author said that there is something to do here
    conv1 = activation(conv1)
    # print("conv1.shape : activation", conv1.shape)

    # Pooling Layer. Input = 28x28x1. Output = 14x14x6.
    # not sure why author says that output is 28x28x1 when I belive it is 28x28x6
    # Valid padding means no padding for sliding windows
    # Data_format = 'NHWC' is default and means Num_samples x Height x Width x Channels
    # 1 for Num_samples and Channels mean that no max_pooling for those -
    # I read somewhere that what matters is shape of the input tensor
    # stride == size of the sliding window (ksize) means that no overlapping of the regions
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # print("pool_1.shape max_pool: ", pool_1.shape)

    # Layer 2: Convolutional. Output = 10x10x16.
    # this time receptive fields is 5x5 and we go for no obvious reason from 6 to 16 channels
    # conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = 0.1))
    conv2_w = tf.Variable(tf.truncated_normal(shape=[receptive_field_2, receptive_field_2, filters_1,
                                                     filters_2], mean=0, stddev=standard_dev))
    # print("conv2_w.shape variable: ", conv2_w.shape)

    # again I am not sure why bias is zero
    conv2_b = tf.Variable(tf.zeros(filters_2))
    # conv2_b = tf.Variable(tf.zeros(16))
    # print("conv2_b.shape zeroes: ", conv2_b.shape)

    # also here is the sliding window is 5x5 and stride of 1 in both axis
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # print("conv2.shape conv2d: ", conv2.shape)

    # Activation.
    conv2 = activation(conv2)
    # print("conv2.shape activation: ", conv2.shape)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    # again ksize == strides means no overlapping sectors when doing pooling
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # print("pool_2.shape max_pool: ", pool_2.shape)

    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    # print("fc1.shape flattened: ", fc1.shape)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    # input needs to be a multiplication of all the dimensions of the input tensor
    # not sure why everywhere here standard deviation is 0.1 rather than 1
    # fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = 0, stddev = 0.1))
    # inf(fc1.shape[1]) casts to int result of the size of the flattened 2nd layer
    # result of the fc1.shape is (?, x) where x is a string or something that needs to be casted to int
    fc1_w = tf.Variable(tf.truncated_normal(shape=(int(fc1.shape[1]), 120), mean=0, stddev=standard_dev))

    # print("shape fc1 : ", fc1.shape)
    # print("shape fc1_w :", fc1_w.shape)

    fc1_b = tf.Variable(tf.zeros(120))
    # fc1_b = tf.Variable(tf.zeros(200))

    # and here comes matmul instead of conv
    # it simply multiplies flatten input tensor by weights and adds bias
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    # Activation.
    fc1 = activation(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0, stddev=standard_dev))
    # fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = 0.1))
    fc2_b = tf.Variable(tf.zeros(84))
    # fc2_b = tf.Variable(tf.zeros(120))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    # Activation.
    fc2 = activation(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=0, stddev=standard_dev))
    # fc3_w = tf.Variable(tf.truncated_normal(shape = (120,10), mean = 0 , stddev = 0.1))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits
