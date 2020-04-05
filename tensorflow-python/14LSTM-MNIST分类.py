# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 输入图片是28*28
n_inputs = 28  # 输入一行，一行有28个数据
max_time = 28  # 一共28行
lstm_size = 100  # 隐层单元
n_classes = 10  # 10个分类
batch_size = 64  # 每批次64个样本
n_batch = mnist.train.num_examples // batch_size  # 计算一共有多少个批次

# 这里的none表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32, [None, 784])

# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))

# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定义RNN网络
def RNN(X, weights, biases):
    # inputs=[batch_size, max_time, n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM
    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
    #    final_state[state, batch_size, cell.state_size]
    #    final_state[0]是cell state
    #    final_state[1]是hidden_state
    #    outputs: The RNN output `Tensor`.
    #       If time_major == False (default), this will be a `Tensor` shaped:
    #         `[batch_size, max_time, cell.output_size]`.
    #       If time_major == True, this will be a `Tensor` shaped:
    #         `[max_time, batch_size, cell.output_size]`.
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


# 计算RNN的返回结果
prediction = RNN(x, weights, biases)

# 损失函数
loss = tf.losses.softmax_cross_entropy(y, prediction)

# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(15):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))
        # acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        # print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))


# Iter 0, Testing Accuracy= 0.8969
# Iter 1, Testing Accuracy= 0.9287
# Iter 2, Testing Accuracy= 0.9548
# Iter 3, Testing Accuracy= 0.9629
# Iter 4, Testing Accuracy= 0.9593
# Iter 5, Testing Accuracy= 0.9678
# Iter 6, Testing Accuracy= 0.9744
# Iter 7, Testing Accuracy= 0.972
# Iter 8, Testing Accuracy= 0.9748
# Iter 9, Testing Accuracy= 0.9699
# Iter 10, Testing Accuracy= 0.9753
