import os 
import tensorflow as tf 

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a,b)
graph = tf.get_default_graph()
print(graph)
var = 2.0
sum2 = a + var
print(sum2)
# placeholder是一个占位符，feed_dict 一个字典
plt = tf.placeholder(tf.float32,[2,3,4])
print(plt)
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(a.shape)
    print(a.graph)
    print("----------")
    print(plt.shape)
    print(a.op)
