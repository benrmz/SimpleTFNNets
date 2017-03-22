
import tensorflow as tf


print ("----------------Beginning NOT net-----------------")
# Truth Table as vectors/matrices, simply output opposite of the single input
not_inputs = [
    [-1.],
    [1.]
]

not_outputs = [
    [1.],
    [-1.],
]

not_weights = tf.Variable(tf.random_normal([1,1]))
not_bias = tf.Variable(tf.zeros([1]))

#acivation function, tanh = scaled version of sigmoid,
hypothesis = tf.tanh(tf.add(tf.matmul( not_inputs, not_weights ) , not_bias))

error = tf.subtract(not_outputs, hypothesis)
cost = tf.reduce_mean(tf.square(error))
# the learning rate, alpha
alpha = 0.05

# using GradientDescentOptimizer as training algorithm, set learning rate
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

#max number of iterations if it fails to converge to target_error
current_error, target_error = 100, 0.001
current_epoch, max_epochs = 0, 1500

#now that we have sup up network, train it
#the weight should become negative, flipping the output
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
            # print ('hypothesis: ', sess.run(hypothesis, feed_dict=zip(and_inputs, and_outputs)))
            print('Weights: ', sess.run(not_weights))
            print('Bias: ', sess.run(not_bias))

    sess.close()
