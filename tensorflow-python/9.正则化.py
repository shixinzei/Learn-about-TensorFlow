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
keep_prob = tf.placeholder(tf.float32)

# 784-1000-500-10
# 创建一个简单的神经网络
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

# 正则项
l2_loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(
    W3) + tf.nn.l2_loss(b3)

# 交叉熵
loss = tf.losses.softmax_cross_entropy(y, prediction) + 0.0005 * l2_loss

# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax:返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))

# Iter 0,Testing Accuracy 0.9429,Training Accuracy 0.94554543
# Iter 1,Testing Accuracy 0.9539,Training Accuracy 0.96009094
# Iter 2,Testing Accuracy 0.9616,Training Accuracy 0.9665091
# Iter 3,Testing Accuracy 0.9613,Training Accuracy 0.96625453
# Iter 4,Testing Accuracy 0.9598,Training Accuracy 0.9628364
# Iter 5,Testing Accuracy 0.9562,Training Accuracy 0.9588364
# Iter 6,Testing Accuracy 0.9647,Training Accuracy 0.9707091
# Iter 7,Testing Accuracy 0.9642,Training Accuracy 0.96867275
# Iter 8,Testing Accuracy 0.963,Training Accuracy 0.96876365
# Iter 9,Testing Accuracy 0.9625,Training Accuracy 0.96738183
# Iter 10,Testing Accuracy 0.9652,Training Accuracy 0.9703818


