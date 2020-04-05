# encoding:utf-8
import tensorflow as tf

# 创建一个常量
m1 = tf.constant([[3, 3]])
print(m1)
print(type(m1))
print(m1.shape)

# 创建一个常量
m2 = tf.constant([[2], [3]])
print(m2)

# 矩阵乘法OP
product = tf.matmul(m1, m2)
print(product)

# 定义会话
sess = tf.Session()

# 调用sess中的run方法来执行矩阵乘法OP
result = sess.run(product)
print(result)
sess.close()

# 这种方法会自动关闭会话
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
