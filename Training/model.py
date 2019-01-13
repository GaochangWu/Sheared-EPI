import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.001)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


def deconv(x, w, output_shape, strides, name=None):
    dyn_input_shape = tf.shape(x)
    batch_size = dyn_input_shape[0]
    output_shape = tf.stack([batch_size, output_shape[1], output_shape[2], output_shape[3]])
    output = tf.nn.conv2d_transpose(x, w, output_shape, strides, padding="SAME", name=name)
    return output


def network(x):
    angRes = 9
    weight = []
    bias = []
    x = tf.transpose(x, [0, 3, 2, 1])
    x = tf.image.resize_bilinear(x, [angRes, 31])
    x = tf.transpose(x, [0, 3, 2, 1])
    # x is input, with size [batch, 31, 31, angRes]
    # Layer 1, convolutional layer
    w = weight_variable([3, 3, 9, 32])
    b = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, w) + b)
    # shape is [batch, 31, 31, 64]
    weight += [w]
    bias += [b]
    h_conv1_pool = max_pool_2x2(h_conv1)
    # shape is [batch, 16, 16, 64]

    # Layer 2, convolutional layer
    w = weight_variable([3, 3, 32, 64])
    b = weight_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_conv1_pool, w) + b)
    # shape is [batch, 16, 16, 64]
    weight += [w]
    bias += [b]
    h_conv2_pool = max_pool_2x2(h_conv2)
    # shape is [batch, 8, 8, 64]

    # Layer 3, convolutional layer
    w = weight_variable([3, 3, 64, 64])
    b = weight_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2_pool, w) + b)
    # shape is [batch, 8, 8, 64]
    weight += [w]
    bias += [b]
    h_conv3_pool = max_pool_2x2(h_conv3)
    # shape is [batch, 4, 4, 64]

    # Layer 4, convolutional layer
    w = weight_variable([3, 3, 64, 64])
    b = weight_variable([64])
    h_conv4 = tf.nn.relu(conv2d(h_conv3_pool, w) + b)
    weight += [w]
    bias += [b]
    # shape is [batch, 4, 4, 64]

    # Layer 5, deconvolutional layer
    w = weight_variable([3, 3, 64, 64])
    b = weight_variable([64])
    h_deconv1 = tf.nn.relu(deconv(h_conv4, w, [-1, 8, 8, 64], [1, 2, 2, 1]) + b)
    weight += [w]
    bias += [b]
    h_cat1 = tf.concat([h_deconv1, h_conv3], 3)
    # shape is [batch, 8, 8, 128]

    # Layer 6, deconvolutional layer
    w = weight_variable([3, 3, 64, 128])
    b = weight_variable([64])
    h_deconv2 = tf.nn.relu(deconv(h_cat1, w, [-1, 16, 16, 64], [1, 2, 2, 1]) + b)
    weight += [w]
    bias += [b]
    h_cat2 = tf.concat([h_deconv2, h_conv2], 3)
    # shape is [batch, 16, 16, 128]

    # Layer 7, deconvolutional layer
    w = weight_variable([3, 3, 32, 128])
    b = weight_variable([32])
    h_deconv3 = tf.nn.relu(deconv(h_cat2, w, [-1, 31, 31, 32], [1, 2, 2, 1]) + b)
    weight += [w]
    bias += [b]
    h_cat3 = tf.concat([h_deconv3, h_conv1], 3)
    # shape is [batch, 31, 31, 64]

    # Layer 8, convolutional layer
    w = weight_variable([3, 3, 64, 1])
    b = weight_variable([1])
    y_conv = conv2d(h_cat3, w) + b
    weight += [w]
    bias += [b]
    # shape is [batch, 31, 31, 1]

    return y_conv, weight, bias
