import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
COST=1
PROFIT = 9

rdm = np.random.RandomState(SEED)#基于seed产生随机数
X = rdm.rand(32,2)#随机数返回300行2列的矩阵，表示300组坐标点
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]#判断如果两个坐标的平方和小于2，给Y赋值1，其余赋值0

#1定义神经网络的输入、参数和输出，定义前向传播过程。
x = tf.placeholder(tf.float32, shape=(None, 2))#占位
y_ = tf.placeholder(tf.float32, shape=(None, 1))#占位
w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))#正态分布
y = tf.matmul(x, w1)#点积

#2定义损失函数及反向传播方法。
#定义损失函数为MSE,反向传播方法为梯度下降。
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y-y_)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()#初始化
    sess.run(init_op)#初始化
    STEPS = 20000#20000轮
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = (i*BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            print ("After %d training steps, w1 is: " % (i))
            print (sess.run(w1), "\n")
    print ("Final w1 is: \n", sess.run(w1))
#在本代码#2中尝试其他反向传播方法，看对收敛速度的影响，把体会写到笔记中

