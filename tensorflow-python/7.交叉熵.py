# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print(mnist)
print(type(mnist))
print(mnist.train.num_examples)
print(mnist.test.num_examples)

# 批次大小
batch_size = 64
# 计算一个周期一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
print(n_batch)

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络:784-10
W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]) + 0.01)  # 偏置值初始化
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数
# 二次代价函数
# loss = tf.losses.mean_squared_error(y, prediction)
# 交叉熵
loss = tf.losses.softmax_cross_entropy(y, prediction)

# # 使用梯度下降法
# train = tf.train.GradientDescentOptimizer(0.35).minimize(loss)
# Adam优化器
train = tf.train.AdamOptimizer(0.002).minimize(loss)

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 周期epoch：所有数据训练一次，就是一个周期
    for epoch in range(51):
        for batch in range(n_batch):
            # 获取一个批次的数据和标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

        # 每训练一个周期做一次测试
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy" + str(acc))
        acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("Iter " + str(epoch) + ",Training Accuracy" + str(acc))

# Iter 50,Testing Accuracy0.9337
# Iter 50,Training Accuracy0.95198184