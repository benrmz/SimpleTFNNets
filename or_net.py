
import tensorflow as tf

######### OR operation ###############

print ("----------------Beginning OR net-----------------")
# Truth Table as vectors/matrices
or_inputs = [
    [1., 1.],
    [1., 0.],
    [0., 1.],
    [0., 0.]
]

or_outputs = [
    [1.],
    [1.],
    [1.],
    [0.]
]
x_ = tf.placeholder(tf.float32, shape=[4,2], name='X-inputs')
y_ = tf.placeholder(tf.float32, shape=[4,1], name='Y-outputs')

or_weights = tf.Variable(tf.random_normal([2,1]))
# or_bias = tf.Variable(tf.zeros([1]))

#acivation function, tanh = scaled version of sigmoid,
hypothesis = tf.tanh(tf.matmul( or_inputs, or_weights ))

error = tf.subtract(or_outputs, hypothesis)
cost = tf.reduce_mean(tf.square(error))
# the learning rate, alpha
alpha = 0.05

# using GradientDescentOptimizer as training algorithm, set learning rate
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

#max number of iterations if it fails to converge to target_error
current_error, target_error = 100, 0.001
current_epoch, max_epochs = 0, 1500

#now that we have sup up network, train it
with tf.Session() as sess:
    #initialze tf vars
    sess.run(tf.global_variables_initializer())

    while current_error > target_error and current_epoch < max_epochs:
        current_epoch += 1
        current_error, t = sess.run([cost, train_step])
        #check the progress of the training:
        if current_epoch % 100 == 0:
            print(" ")
            print( 'CHECKING PROGRESS')
            print('epoch: ', current_epoch)
            print('cost: ', sess.run(cost))
            print ('hypothesis: ', sess.run(hypothesis, feed_dict={x_: or_inputs, y_: or_outputs}))
            print('Weights: ', sess.run(or_weights))
            # print('Bias: ', sess.run(or_bias))

    sess.close()
