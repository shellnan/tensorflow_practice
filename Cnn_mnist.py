import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0))
    return w

def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0,shape=shape))
    return b

def model():

    # 1.x [None,784] y_true[None,10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32,[None,784])
        y_true = tf.placeholder(tf.int32,[None,10])
    # 2. filter:5*5*1,32个,strides = 1 激活：tf.nn.relu 池化
    with tf.variable_scope("conv1"):
        w_conv1 = weight_variables([5,5,1,32])
        b_conv1 = bias_variables([32])
        x_reshape = tf.reshape(x,[-1,28,28,1])
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape,w_conv1,strides=[1,1,1,1],padding="SAME")+b_conv1)
        x_pool1 = tf.nn.max_pool(x_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    with tf.variable_scope("conv2"):
        w_conv2 = weight_variables([5,5,32,64])
        b_conv2 = bias_variables([64])
        # [None,14,14,32]---->[None,14,14,64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1,w_conv2,strides=[1,1,1,1],padding="SAME")+b_conv2)
        x_pool2 = tf.nn.max_pool(x_relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    with tf.variable_scope("fc"):
        w_fc = weight_variables([7*7*64,10])
        b_fc = bias_variables([10])
        x_fc_reshape = tf.reshape(x_pool2,[-1,7*7*64])
        y_predict = tf.matmul(x_fc_reshape,w_fc)+b_fc
    return x,y_true,y_predict
def conv_fc():
    mnist = input_data.read_data_sets("./input_data",one_hot=True)
    x,y_true,y_predict = model()
    with tf.variable_scope("soft_cross"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(10000):
            mnist_x, mnist_y = mnist.train.next_batch(50)
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
            print("训练第%d步,准确率为:%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))
    return None

if __name__=="__main__":
    conv_fc()
