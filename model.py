from __future__ import division
import tensorflow as tf
import numpy as np
#import data

#THE ACCURACY EVALUATION FUNCTION
def get_accuracy(net_out, answers):
    outputs = np.round(net_out)
    return np.sum(answers == outputs)

x = tf.placeholder(tf.float32, [None, 80*120])
y_ = tf.placeholder(tf.float32, [None, 1])

x_reshaped = tf.reshape(x, [-1, 80, 120, 1])
#now it's 80 x 120

W1_convo = tf.Variable(tf.truncated_normal([23,23,1,35]), name="Wc1")
b1_convo = tf.Variable(tf.zeros([35]), name="bc1")
convo1 = tf.nn.relu(tf.nn.conv2d(x_reshaped, W1_convo, strides=[1,1,1,1], padding="VALID") + b1_convo)
#now it's 58 x 98
pool1 = tf.nn.max_pool(convo1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
#now it's 29 x 49

W2_convo = tf.Variable(tf.truncated_normal([10,10,35,55]), name="Wc2")
b2_convo = tf.Variable(tf.zeros([55]), name="bc2")
convo2 = tf.nn.relu(tf.nn.conv2d(pool1, W2_convo, strides=[1,1,1,1], padding="VALID") + b2_convo)
#now it's 20 * 40
pool2 = tf.nn.max_pool(convo2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
#now it's 10 * 20

'''
W3_convo = tf.Variable(tf.truncated_normal([3,3,48,48]))
b3_convo = tf.Variable(tf.zeros([48]))
convo3 = tf.nn.relu(tf.nn.conv2d(pool2, W3_convo, strides=[1,1,1,1], padding="VALID") + b3_convo)
#now it's 8 * 18
pool3 = tf.nn.max_pool(convo3, ksize=[1,2,2,1], strides=[1,1,1,1], padding="VALID")
#now it's 7 * 17
'''

flattened = tf.reshape(pool2, [-1, 10*20*55])

W3 = tf.Variable(tf.truncated_normal([10*20*55, 512]), name="W3")
b3 = tf.Variable(tf.zeros([512]), name="b3")

FC3 = tf.nn.sigmoid(tf.matmul(flattened, W3) + b3)

W4 = tf.Variable(tf.truncated_normal([512, 1]), name="W4")
b4 = tf.Variable(tf.zeros([1]), name="b4")

y = tf.nn.sigmoid(tf.matmul(FC3, W4) + b4)

'''
cost = tf.reduce_mean(tf.pow((y-y_),2))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
init = tf.global_variables_initializer()
sess.run(init)
'''
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "saves/vars.ckpt")

def run(img):
	return sess.run(y, feed_dict={x:img.reshape((1,9600))})

'''
for e in xrange(500):
    for i in xrange(200):
        xs, ys = data.get_batch(8)
        sess.run(train_step, feed_dict={x:xs, y_:ys})
    xs_te, ys_te = data.get_test_data()
    results = sess.run(y, feed_dict={x:xs_te})
    print "EPOCH {} --> {}".format(e, get_accuracy(results, ys_te))
    if e % 10 == 0:
        big_results1 = sess.run(y, feed_dict={x:data.training_images[:800]})
        big_results1 = np.round(big_results1)
        s1 = np.sum(big_results1 == data.training_labels[:800])
        big_results2 = sess.run(y, feed_dict={x:data.training_images[800:1600]})
        big_results2 = np.round(big_results2)
        s2 = np.sum(big_results2 == data.training_labels[800:1600])
        big_results3 = sess.run(y, feed_dict={x:data.training_images[1600:]})
        big_results3 = np.round(big_results3)
        s3 = np.sum(big_results3 == data.training_labels[1600:])
        accuracy = 100.0 * (s1 + s2 + s3) / len(data.training_labels)
        print "TOTAL ACCURACY = {}%".format(accuracy)
        if accuracy > 90:
            break

#save variables
saver.save(sess, "saves/vars.ckpt")
print "MODEL SAVED"
'''
