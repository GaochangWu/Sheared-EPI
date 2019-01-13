import tensorflow as tf
import numpy as np
import os
import model3_2 as model
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------Parameters setting-------------------- #
modelSavePath = "../Model/model.ckpt"
matSavePath = '../Model/model.mat'

x = tf.placeholder(tf.float32, shape=[None, 31, 31, 2])

y_conv, W, B = model.network(x)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, modelSavePath)
    W, B = sess.run([W, B])
    print("Model restored.")
    scio.savemat(matSavePath, {
        'weight': W,
        'bias': B
    }, format='5'
    )