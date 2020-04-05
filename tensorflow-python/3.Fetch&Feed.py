# encoding:utf-8
import tensorflow as tf

# Fetch：可以在session中同时计算多个tensor或执行多个操作

# 定义三个常量
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

# 加法op
add = tf.add(input2, input3)

# 乘法op
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result1, result2 = sess.run([mul, add])
    print(result1, result2)

# Feed:先定义占位符，等需要的时候再传入数据
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 乘法op
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: 8.0, input2: 2.0}))

