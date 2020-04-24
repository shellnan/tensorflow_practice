import tensorflow as tf
import os

Q = tf.FIFOQueue(3,tf.float32)

enq_many = Q.enqueue_many([[0.1,0.2,0.3],])

out_q = Q.dequeue()
data = out_q + 1

en_q = Q.enqueue(data)

with tf.Session() as sess:
    # 初始化队列
    sess.run(enq_many)
    # 处理数据
    for i in range(100):
        sess.run(en_q)

    for i in range(Q.size().eval()):
        print(sess.run(Q.dequeue()))

