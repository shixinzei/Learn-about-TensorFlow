# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 64
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵代价函数
# loss = tf.losses.softmax_cross_entropy(y,prediction)
# mse
loss = tf.losses.mean_squared_error(y, prediction)

# 使用梯度下降法
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# Adam优化器
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax:返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))

# Iter 0,Testing Accuracy 0.9114,Training Accuracy 0.9083091
# Iter 1,Testing Accuracy 0.9218,Training Accuracy 0.9195273
# Iter 2,Testing Accuracy 0.9266,Training Accuracy 0.92454547
# Iter 3,Testing Accuracy 0.9272,Training Accuracy 0.928
# Iter 4,Testing Accuracy 0.9287,Training Accuracy 0.9300909
# Iter 5,Testing Accuracy 0.9289,Training Accuracy 0.9323273
# Iter 6,Testing Accuracy 0.9309,Training Accuracy 0.9338545
# Iter 7,Testing Accuracy 0.9307,Training Accuracy 0.9347636
# Iter 8,Testing Accuracy 0.9313,Training Accuracy 0.9369636
# Iter 9,Testing Accuracy 0.9331,Training Accuracy 0.93707275
# Iter 10,Testing Accuracy 0.932,Training Accuracy 0.93792725
# Iter 11,Testing Accuracy 0.9325,Training Accuracy 0.93845457
# Iter 12,Testing Accuracy 0.932,Training Accuracy 0.93796366
# Iter 13,Testing Accuracy 0.9318,Training Accuracy 0.9396909
# Iter 14,Testing Accuracy 0.9338,Training Accuracy 0.94067276
# Iter 15,Testing Accuracy 0.9335,Training Accuracy 0.9414182
# Iter 16,Testing Accuracy 0.9323,Training Accuracy 0.94192725
# Iter 17,Testing Accuracy 0.9322,Training Accuracy 0.94247276
# Iter 18,Testing Accuracy 0.9331,Training Accuracy 0.9425091
# Iter 19,Testing Accuracy 0.9325,Training Accuracy 0.94274545
# Iter 20,Testing Accuracy 0.9333,Training Accuracy 0.9430909

