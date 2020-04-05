# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print(mnist)
print(type(mnist))
print(mnist.train.num_examples)
print(mnist.test.num_examples)

# 批次大小
batch_size = 64

# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
print(n_batch)

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络:784-10
W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.01))  # seddev标准差
b = tf.Variable(tf.zeros([10]) + 0.1)  # 偏置值初始化
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数 mse 均方差
loss = tf.losses.mean_squared_error(y, prediction)
# 优化器 使用梯度下降法 0.6的学习率
train = tf.train.GradientDescentOptimizer(0.6).minimize(loss)
# train = tf.train.AdamOptimizer(0.002).minimize(loss)

# 独热编码
# 1->0100000000
# 2->0010000000
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率 true->1 false->0
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # for _ in range(1000):
    # 周期epoch：所有的数据训练一次，就是一个周期
    for epoch in range(71):
        for batch in range(n_batch):
            # 获取一个批次的数据和标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

        # 每训练一个周期做一次测试
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy" + str(acc))
        acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("Iter " + str(epoch) + ",Training Accuracy" + str(acc))


# Iter 70,Testing Accuracy0.9301
# Iter 70,Training Accuracy0.93523633
