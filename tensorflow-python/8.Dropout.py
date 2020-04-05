# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# # 载入数据集
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#
# # 每个批次的大小
# batch_size = 64
# # 计算一共有多少个批次
# n_batch = mnist.train.num_examples // batch_size
#
# # 定义三个placeholder
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
# keep_prob = tf.placeholder(tf.float32)
#
# # 网络结构：784-1000-500-10
# W1 = tf.Variable(tf.truncated_normal([784, 1000], stddev=0.1))
# b1 = tf.Variable(tf.zeros([1000]) + 0.1)
# L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# L1_drop = tf.nn.dropout(L1, keep_prob)
#
# W2 = tf.Variable(tf.truncated_normal([1000, 500], stddev=0.1))
# b2 = tf.Variable(tf.zeros([500]) + 0.1)
# L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
# L2_drop = tf.nn.dropout(L2, keep_prob)
#
# W3 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
# b3 = tf.Variable(tf.zeros([10]) + 0.1)
# prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)
#
# # 交叉熵
# loss = tf.losses.softmax_cross_entropy(y, prediction)
# # 使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#
# # 初始化变量
# init = tf.global_variables_initializer()
#
# # 结果存放在一个布尔型列表中
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# # 求准确率
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(31):
#         for batch in range(n_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
#
#         test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
#         train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
#         print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))

# Iter 29,Testing Accuracy 0.9663,Training Accuracy 0.97196364
# Iter 30,Testing Accuracy 0.9671,Training Accuracy 0.9722546

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 64
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义三个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 网络结构：784-1000-500-10
W1 = tf.Variable(tf.truncated_normal([784, 1000], stddev=0.1))
b1 = tf.Variable(tf.zeros([1000]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([1000, 500], stddev=0.1))
b2 = tf.Variable(tf.zeros([500]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 交叉熵
loss = tf.losses.softmax_cross_entropy(y, prediction)
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))

# Iter 0,Testing Accuracy 0.9504,Training Accuracy 0.9533273
# Iter 1,Testing Accuracy 0.9616,Training Accuracy 0.9692364
# Iter 2,Testing Accuracy 0.9645,Training Accuracy 0.9778
# Iter 3,Testing Accuracy 0.9702,Training Accuracy 0.98212725
# Iter 4,Testing Accuracy 0.9728,Training Accuracy 0.98605454
# Iter 5,Testing Accuracy 0.9702,Training Accuracy 0.9851273
# Iter 6,Testing Accuracy 0.9747,Training Accuracy 0.9892182
# Iter 7,Testing Accuracy 0.9748,Training Accuracy 0.99023634
# Iter 8,Testing Accuracy 0.9773,Training Accuracy 0.9914182
# Iter 9,Testing Accuracy 0.9771,Training Accuracy 0.9920727
# Iter 10,Testing Accuracy 0.9781,Training Accuracy 0.99256366
# Iter 11,Testing Accuracy 0.9778,Training Accuracy 0.99292725
# Iter 12,Testing Accuracy 0.9792,Training Accuracy 0.99325454
# Iter 13,Testing Accuracy 0.9803,Training Accuracy 0.99365455
# Iter 14,Testing Accuracy 0.9797,Training Accuracy 0.99405456
# Iter 15,Testing Accuracy 0.9799,Training Accuracy 0.99436367
# Iter 16,Testing Accuracy 0.9805,Training Accuracy 0.9945091
# Iter 17,Testing Accuracy 0.9815,Training Accuracy 0.99472725
# Iter 18,Testing Accuracy 0.981,Training Accuracy 0.995
# Iter 19,Testing Accuracy 0.9815,Training Accuracy 0.9951636
# Iter 20,Testing Accuracy 0.9807,Training Accuracy 0.9954
# Iter 21,Testing Accuracy 0.9811,Training Accuracy 0.9955636
# Iter 22,Testing Accuracy 0.9804,Training Accuracy 0.9956909
# Iter 23,Testing Accuracy 0.9818,Training Accuracy 0.9958909
# Iter 24,Testing Accuracy 0.9815,Training Accuracy 0.99596363
# Iter 25,Testing Accuracy 0.9809,Training Accuracy 0.9960909
# Iter 26,Testing Accuracy 0.9815,Training Accuracy 0.99610907
# Iter 27,Testing Accuracy 0.981,Training Accuracy 0.99612725
# Iter 28,Testing Accuracy 0.9808,Training Accuracy 0.99618185
# Iter 29,Testing Accuracy 0.9814,Training Accuracy 0.9962
# Iter 30,Testing Accuracy 0.9814,Training Accuracy 0.9962182



