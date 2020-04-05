# encoding:utf-8
import tensorflow as tf

# 定义一个变量
x = tf.Variable([1, 2])

# 定义一个常量
a = tf.constant([3, 3])

# 减法op
sub = tf.subtract(x, a)

# 加法op
add = tf.add(x, sub)

# 所有变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 执行变量初始化
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

