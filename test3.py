import tensorflow as tf

def myregression():
    """
    实现线性回归
    """
# 1 准备数据，x [100,1] ,y 目标值[100]
    x =tf.random_normal([100,1],mean=1.75,stddev=0.5,name="x_data")
# 矩阵相乘必须是二维的
    y_true = tf.matmul(x,[[0.7]]) +0.8
# 2 建立线性回归模型 1个特征，1个权重，一个偏置
    # 用变量定义才能优化
    weight = tf.Variable(tf.random_normal([1,1],mean=0.0,stddev=1.0),name="weight")
    bias = tf.Variable(0.0,name="b")
    y_predict=tf.matmul(x,weight) + bias
# 3 建立损失函数，均方误差
    loss = tf.reduce_mean(tf.square(y_true-y_predict))
# 4 梯度下降优化损失 

    train_op = tf.train.GradientDescentOptimizer(0.1186).minimize(loss)
    # 定义一个初始化变量的op
    #收集tensor
    tf.summary.scalar("losses",loss)
    tf.summary.histogram("weights",weight)
    # 定义合并tensor的op
    merged = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        
        print("权重为： %f,偏置为：%f"%(weight.eval(),bias.eval()))
        filewriter = tf.summary.FileWriter("../graph_note",graph=sess.graph)
        for i in range(1000):
            sess.run(train_op)
            summary = sess.run(merged)
            filewriter.add_summary(summary,i)
            print("权重为： %f,偏置为：%f"%(weight.eval(),bias.eval()))

    return None
if __name__=="__main__":
    myregression()
