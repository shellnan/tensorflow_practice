import tensorflow as tf

x = tf.constant([[0.7,0.5]])
with tf.variable_scope("w"):

    w1 =tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
tf.summary.histogram("weight",w1)
tf.summary.histogram("weight2",w2)
merged = tf.summary.merge_all()
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    filewriter = tf.summary.FileWriter("./",graph=sess.graph)
    sess.run(init_op)
    print("y in is",sess.run(y))
    saver.save(sess,"./line_model")
