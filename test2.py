import tensorflow as tf

a = tf.constant([1,2,3,4,5])
var = tf.Variable(tf.random_normal([2,3],mean=0,stddev=1.0))

print(a,var)
# 必须做一步显示的初始化op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 必须运行初始化op
    sess.run(init_op)
    print(sess.run([a,var]))
# plt = tf.placeholder(tf.float32,[None,2])

# print(plt)

# plt.set_shape([3,2])

# print(plt)

# with tf.Session() as sess:
#    pass
