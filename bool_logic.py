# Benjamin Ramirez 3/18/2017
# training neural networks to perform logical operations using tensorflow
# objective is to demonstrate/learn tensorflow library

# following network architecture from this site:
# http://toritris.weebly.com/perceptron-2-logical-operations.html
# and elementary intro to tensorflow here: https://medium.com/@jaschaephraim/elementary-neural-networks-with-tensorflow-c2593ad3d60b#.brk1wtlif

import tensorflow as tf

########### AND operation ###############

# Truth Table as vectors/matrices
and_inputs = [
    [1., 1.],
    [1., -1.],
    [-1., 1.],
    [-1., -1.]
]

and_outputs = [
    [1.],
    [-1.],
    [-1.],
    [-1.]
]

x_ = tf.placeholder(tf.float32, shape=[4,2], name='X-inputs')
y_ = tf.placeholder(tf.float32, shape=[4,1], name='Y-outputs')

and_weights = tf.Variable(tf.random_normal([2,1]))
and_bias = tf.Variable(tf.zeros([1]))

#acivation function, tanh = scaled version of sigmoid,
hypothesis = tf.tanh(tf.add(tf.matmul( and_inputs, and_weights ) , and_bias))

error = tf.subtract(and_outputs, hypothesis)
cost = tf.reduce_mean(tf.square(error))
# the learning rate, alpha
alpha = 0.05

# using GradientDescentOptimizer as training algorithm, set learning rate
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

#max number of iterations if it fails to converge to target_error
current_error, target_error = 100, 0.001
current_epoch, max_epochs = 0, 10000

with tf.Session() as sess:
    #initialze tf vars
    sess.run(tf.global_variables_initializer())

    while current_error > target_error and current_epoch < max_epochs:
        current_epoch += 1
        current_error, t = sess.run([cost, train_step])
        #check the progress of the training:
        if current_epoch % 100 == 0:
            print()
            print( 'CHECKING PROGRESS')
            print('epoch: ', current_epoch)
            print('cost: ', sess.run(cost))
            print ('hypothesis: ', sess.run(hypothesis, feed_dict={x_:and_inputs, y_:and_outputs}))
            print('Weights: ', sess.run(and_weights))
            print('Bias: ', sess.run(and_bias))

    sess.close()


