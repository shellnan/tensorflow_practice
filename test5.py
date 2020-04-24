import tensorflow as tf 
# 定义队列
Q = tf.FIFOQueue(1000,tf.float32)
# 变量
var = tf.Variable(0.0)
# 实现一个自增 tf.assign_add
data = tf.assign_add(var,tf.constant(1.0))

en_q = Q.enqueue(data)
# 定义队列管理器op，指定多少个子线程
qr = tf.train.QueueRunner(Q,enqueue_ops=[en_q]*2)
# 初始化变量的op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)
    # 开启线程管理器
    coord = tf.train.Coordinator()

    threads = qr.create_threads(sess,coord=coord,start=True)

    for i in range(300):
        print(sess.run(Q.dequeue()))

    coord.request_stop()

    coord.join(threads)
