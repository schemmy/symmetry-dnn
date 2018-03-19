import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import random
import math
import matplotlib.pyplot as plt


random.seed(3)

def rand_cluster(n,c,r):
    """returns n random points in disk of radius r centered at c"""
    x,y = c
    points = []
    for i in range(n):
        theta = 2*math.pi*random.random()
        s = r*random.random()
        points.append((x+s*math.cos(theta), y+s*math.sin(theta)))
    return points

def rand_clusters(k,n,r, a,b,c,d):
    """return k clusters of n points each in random disks of radius r
    where the centers of the disk are chosen randomly in [a,b]x[c,d]"""
    clusters = []
    for _ in range(k):
        x = a + (b-a)*random.random()
        y = c + (d-c)*random.random()
        clusters.extend(rand_cluster(n,(x,y),r))
    return clusters

n = 50
X = rand_clusters(2,50,1.8,-1,1,-1,1)
data = np.array(X)
label = np.transpose(np.array([[1]*n + [0]*n, [0]*n + [1]*n]))
# label = np.array([1]*n + [0]*n)
# print (data, label)

# plt.scatter(data[:n,0], data[:n,1], color=['red'])
# plt.scatter(data[n:,0], data[n:,1], color=['green'])
# plt.show()


def weight_variable(shape, name):
    # initial = tf.truncated_normal(shape=shape, stddev=0.1)
    initial = tf.constant(0., shape=shape)
    return tf.get_variable(name=name, initializer=initial)

def bias_variable(shape, name):
    initial = tf.constant(0., shape=shape)
    return tf.get_variable(name=name, initializer=initial)


x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 2])

#hidden layer
# W_fc1 = weight_variable([2, 4], 'W1')
# b_fc1 = bias_variable([4], 'b1')
# h_fc1 = tf.sigmoid(tf.matmul(x, W_fc1) + b_fc1)
# #output layer
# W_fc2 = weight_variable([4, 2], 'W2')
# b_fc2 = bias_variable([2], 'b2')
# y = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)


n_input = 2
n_hidden = 3
n_output = 2
# lmd = 1e-4
lmd = 0#e-4
# parameters = tf.Variable(tf.concat([tf.truncated_normal([n_input * n_hidden]), tf.zeros([n_hidden]),\
                            # tf.truncated_normal([n_hidden * n_output]), tf.zeros([n_output])],0))
parameters = tf.Variable(tf.concat([tf.truncated_normal([n_input * n_hidden]), tf.zeros([n_hidden]),\
                            tf.truncated_normal([n_hidden * n_output]), tf.zeros([n_output])], 0))

idx_from = 0 
weights1 = tf.reshape(tf.slice(parameters, begin=[idx_from], size=[n_input*n_hidden]), [n_input, n_hidden])
idx_from = idx_from + n_input*n_hidden
biases1 = tf.reshape(tf.slice(parameters, begin=[idx_from], size=[n_hidden]), [n_hidden])
hidden = tf.sigmoid(tf.matmul(x, weights1) + biases1)

idx_from = idx_from + n_hidden
weights2 = tf.reshape(tf.slice(parameters, begin=[idx_from], size=[n_hidden*n_output]), [n_hidden, n_output])
idx_from = idx_from + n_hidden*n_output
biases2 = tf.reshape(tf.slice(parameters, begin=[idx_from], size=[n_output]), [n_output]) 
y = tf.nn.sigmoid(tf.matmul(hidden, weights2) + biases2)

weights = tf.concat([tf.reshape(weights1, [-1]), tf.reshape(weights2, [-1])], 0)
regularizer = tf.nn.l2_loss(weights)

los = tf.reduce_mean(tf.reduce_sum(tf.pow(y_ - y, 2), reduction_indices=[1])) #I also tried simply tf.nn.l2_loss(y_ - y)
loss = los + lmd * regularizer

optimizer = tf.train.AdamOptimizer(1.)
grads_and_vars = optimizer.compute_gradients(loss)
hess = tf.hessians(loss, parameters)
train_step = optimizer.apply_gradients(grads_and_vars)

def get_accuracy():
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(accuracy, feed_dict={x: data, y_: label})

def get_norm_grad():
    nng = 0.
    for gv in grads_and_vars:
        # print(str(sess.run(gv[0], feed_dict={x: data, y_: label})) + " - " + gv[1].name)
        grad = sess.run(gv[0], feed_dict={x: data, y_: label})
        nng += np.linalg.norm(grad[0]) ** 2
    return np.sqrt(nng)

def display(w):

    idx_from = 0 
    weights1 = np.reshape(w[idx_from: n_input*n_hidden], [n_input, n_hidden])
    idx_from = idx_from + n_input*n_hidden
    biases1 = np.reshape(w[idx_from: idx_from+n_hidden], [n_hidden])
    idx_from = idx_from + n_hidden
    weights2 = np.reshape(w[idx_from: idx_from+n_hidden*n_output], [n_hidden, n_output])
    idx_from = idx_from + n_hidden*n_output
    biases2 = np.reshape(w[idx_from: idx_from+n_output], [n_output])
    print (weights1)
    print (weights2)

sess = tf.InteractiveSession()


dic = {}
for _ in range(20):

    tf.global_variables_initializer().run()
    flag = 0
    for i in range(2000):
        sess.run(train_step, feed_dict={x: data, y_: label})
        nng = get_norm_grad()
        if nng < 1e-8:
            flag = 1
            break

    if flag == 1:
        v, H, w = sess.run([loss, hess, parameters], feed_dict={x: data, y_: label})    
        eigs = sorted(np.linalg.eigvals(H)[0])
        # print (eigs)
        print("Epoch {}, accuracy {:.2f}%, loss {:.6f}, nng {:.4g}, nnw {:.4g}, high_eig {:.4g}, low_eig {:.4g}."\
                    .format(i+1, get_accuracy()*100, v, nng, np.linalg.norm(w[:4]), max(eigs), min(eigs) ))
        # display(w)
        dic[int(v * 1e4)] = dic.get(int(v * 1e4), []) + [w]

