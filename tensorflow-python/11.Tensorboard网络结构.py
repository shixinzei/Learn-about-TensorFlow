# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 批次大小
batch_size = 64
# 计算一个周期一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer'):
    # 创建一个简单的神经网络:784-10
    with tf.name_scope('weights'):
        W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]) + 0.1)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 二次代价函数
    loss = tf.losses.mean_squared_error(y, prediction)
with tf.name_scope('train'):
    # 使用梯度下降法
    train = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/', sess.graph)
    # 周期epoch：所有数据训练一次，就是一个周期
    for epoch in range(21):
        for batch in range(n_batch):
            # 获取一个批次的数据和标签
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: batch_xs, y: batch_ys})

        # 每训练一个周期做一次测试
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))

# Iter 0,Testing Accuracy 0.856,Training Accuracy 0.84674543
# Iter 1,Testing Accuracy 0.8829,Training Accuracy 0.8739091
# Iter 2,Testing Accuracy 0.893,Training Accuracy 0.8848909
# Iter 3,Testing Accuracy 0.8993,Training Accuracy 0.8918545
# Iter 4,Testing Accuracy 0.9023,Training Accuracy 0.8970182
# Iter 5,Testing Accuracy 0.9042,Training Accuracy 0.89976364
# Iter 6,Testing Accuracy 0.9065,Training Accuracy 0.90278184
# Iter 7,Testing Accuracy 0.9086,Training Accuracy 0.9051818
# Iter 8,Testing Accuracy 0.912,Training Accuracy 0.9067818
# Iter 9,Testing Accuracy 0.9115,Training Accuracy 0.9084
# Iter 10,Testing Accuracy 0.9144,Training Accuracy 0.9099454
# Iter 11,Testing Accuracy 0.915,Training Accuracy 0.9111091
# Iter 12,Testing Accuracy 0.9151,Training Accuracy 0.9122546
# Iter 13,Testing Accuracy 0.9155,Training Accuracy 0.9136
# Iter 14,Testing Accuracy 0.9155,Training Accuracy 0.9139636
# Iter 15,Testing Accuracy 0.9171,Training Accuracy 0.9146182
# Iter 16,Testing Accuracy 0.9173,Training Accuracy 0.91556364
# Iter 17,Testing Accuracy 0.918,Training Accuracy 0.9164
# Iter 18,Testing Accuracy 0.9177,Training Accuracy 0.91712725
# Iter 19,Testing Accuracy 0.9189,Training Accuracy 0.91778183
# Iter 20,Testing Accuracy 0.919,Training Accuracy 0.9180545
