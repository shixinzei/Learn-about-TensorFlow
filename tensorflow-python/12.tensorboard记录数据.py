# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 64
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # 平均值
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 标准差
        tf.summary.scalar('stddev', stddev)
        # 最大值
        tf.summary.scalar('max', tf.reduce_max(var))
        # 最小值
        tf.summary.scalar('min', tf.reduce_min(var))
        # 直方图
        tf.summary.histogram('histogram', var)


# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer'):
    # 创建一个简单的神经网络
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 二次代价函数
    loss = tf.losses.mean_squared_error(y, prediction)
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})

        writer.add_summary(summary, epoch)
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))

# Iter 47,Testing Accuracy 0.9268,Training Accuracy 0.9276182
# Iter 48,Testing Accuracy 0.9263,Training Accuracy 0.9277818
# Iter 49,Testing Accuracy 0.9272,Training Accuracy 0.92785454
# Iter 50,Testing Accuracy 0.9278,Training Accuracy 0.9281818


# with tf.Session() as sess:
#     sess.run(init)
#     writer = tf.summary.FileWriter('logs/', sess.graph)
#     epoch = 0
#     for batch in range(10001):
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
#
#         writer.add_summary(summary, batch)
#         if batch % 1000 == 0:
#             epoch = epoch + 1
#             test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
#             train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
#             print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))

# Iter 1,Testing Accuracy 0.2685,Training Accuracy Iter 1,Testing Accuracy 0.3463,Training Accuracy 0.3420909
# Iter 2,Testing Accuracy 0.8785,Training Accuracy 0.86752725
# Iter 3,Testing Accuracy 0.8955,Training Accuracy 0.8861273
# Iter 4,Testing Accuracy 0.9029,Training Accuracy 0.89483637
# Iter 5,Testing Accuracy 0.9056,Training Accuracy 0.89901817
# Iter 6,Testing Accuracy 0.9086,Training Accuracy 0.9031818
# Iter 7,Testing Accuracy 0.9113,Training Accuracy 0.90574545
# Iter 8,Testing Accuracy 0.913,Training Accuracy 0.90754545
# Iter 9,Testing Accuracy 0.9144,Training Accuracy 0.9096182
# Iter 10,Testing Accuracy 0.9154,Training Accuracy 0.91107273
# Iter 11,Testing Accuracy 0.9174,Training Accuracy 0.9127091

