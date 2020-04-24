import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
#print(X)
Y_ = [[int(x0+x1<1)]for (x0,x1) in X]
print("Y",Y_)
x= tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1)) # 正态分布随机数
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1)) # 正态分布随机数
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

loss_mse = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 初始化
    sess.run(init_op)
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))

    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32 #i*8%32
        end = start + BATCH_SIZE 
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("w1:\n", w1.eval())
    print("w2:\n", w2.eval())
