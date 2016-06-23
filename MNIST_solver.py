
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Session() as sess:
#placeholfers. x - for iamge. y - output class. y consist a 2d tensor, each row is a one-hot 10-dim vectore.
    #shape is optional - allow to catch bugs.
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    #the weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    #init
    sess.run(tf.initialize_all_variables())
    #apply softmax
    y = tf.nn.softmax(tf.matmul(x,W)+b)

    #CE loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    #GD
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #feed data batch
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))