import tensorflow as tf
import os
import model as model
import readH5
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------Parameters setting-------------------- #
trainDataPath = "./training_data.h5" # Prepare your .h5 training data
modelLoadPath = "../Model/model.ckpt"
modelSavePath = "../Model/model.ckpt"
input_channel = 5
res_spatial = 31
res_angular = 2

batchSize = 20
batchSizeTest = 1000
learningRateBase = 0.0001
learningRateDecay = 0.99
trainingIters = 800000
stepDisplay = 100
stepTest = 1000

# ---------Define training pairs and the network -------------#
x = tf.placeholder(tf.float32, shape=[None, res_spatial, res_spatial, res_angular])
y_ = tf.placeholder(tf.float32, shape=[None, res_spatial, res_spatial, 1])
global_step = tf.Variable(0, trainable=False)
keep_prob = tf.placeholder("float")

y_conv, W, B = model.network(x)

# --------------------------Compute loss------------------------------#
loss = tf.reduce_mean(tf.square(y_ - y_conv))

train_step = tf.train.AdamOptimizer(learningRateBase).minimize(loss, global_step=global_step)

# ------------------------Prepare .H5 data------------------------------#
data, label = readH5.read_training_data(trainDataPath)
data_list = [i for i in range(data.shape[0])]
random.shuffle(data_list)
data = data[data_list]
label = label[data_list]
test_data = data[data.shape[0]-batchSizeTest:data.shape[0]]
test_label = label[data.shape[0]-batchSizeTest:data.shape[0]]

train_data = data[0:data.shape[0]-batchSizeTest]
train_label = label[0:data.shape[0]-batchSizeTest]
data_size = int(train_data.shape[0] / batchSize)
print("Training data number is %d, and batch size is %d. Test data number is %d."
      % (data_size*batchSize, batchSize, test_data.shape[0]))

# ------------------------Start training------------------------------#
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=1)
    gstep = 0
    # Try to restore weights from previous training
    try:
        saver_previous = tf.train.Saver(max_to_keep=1)
        saver_previous = saver_previous.restore(sess, modelLoadPath)
    except:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=1)
        flag_retrain = 0
        print('Restoring previous trained weights failed, starting a new training opt.')
    else:
        saver = tf.train.Saver(max_to_keep=1)
        flag_retrain = 1
        gstep = sess.run(global_step)
        print('Restoring previous trained weights, continue the training opt from global step', gstep, '.')

    minLoss = 100
    maxAcc = 0
    epoch = 0
    for i in range(gstep, trainingIters):
        batch_data = train_data[(i % data_size) * batchSize: ((i + 1) % data_size) * batchSize, :, :, :]
        batch_label = train_label[(i % data_size) * batchSize: ((i + 1) % data_size) * batchSize, :]
        if i % data_size == 0:
            epoch += 1
        if i % stepDisplay == 0:
            curLoss = sess.run(loss, feed_dict={x: batch_data, y_: batch_label})
            if i % stepTest == 0:
                curLossTest = sess.run(loss, feed_dict={x: test_data, y_: test_label})
                print("Epoch %2.0f, iteration %6.0f: batch Loss = %.5f for training data, "
                      "test Loss = %.5f for test data." % (epoch, i, curLoss, curLossTest))
                if curLossTest < minLoss:
                    minLoss = curLossTest
                    saver.save(sess, modelSavePath)
            else:
                print("Epoch %2.0f, iteration %6.0f: batch Loss = %.5f for training data." % (epoch, i, curLoss))
        sess.run(train_step, feed_dict={x: batch_data, y_: batch_label})

print("Optimization Finished!")
