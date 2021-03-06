
import tensorflow as tf

########### NAND operation ###############

# Truth Table as input X, output y
nand_inputs = [
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
]

nand_outputs = [
    [1.],
    [1.],
    [1.],
    [0.]
]

x_ = tf.placeholder(tf.float32, shape=[4,2], name='X-inputs')
y_ = tf.placeholder(tf.float32, shape=[4,1], name='Y-outputs')

#theta1 operates on input layer, theta2 from hidden layer to output layer
theta1 = tf.Variable(tf.random_normal([2,2]))
theta2 = tf.Variable(tf.random_normal([2,1]))

bias1 = tf.Variable(tf.zeros([2]))
bias2 = tf.Variable(tf.zeros([1]))


#acivation function, tanh = scaled version of sigmoid,
hidden_ouput = tf.tanh(tf.add( tf.matmul( nand_inputs, theta1 ), bias1))
hypothesis = tf.tanh(tf.add( tf.matmul(hidden_ouput, theta2), bias2))

error = tf.subtract(nand_outputs, hypothesis)
cost = tf.reduce_mean(tf.square(error))
# the learning rate, alpha
alpha = 0.05

# using GradientDescentOptimizer as training algorithm, set learning rate
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

#max number of iterations if it fails to converge to target_error
current_error, target_error = 100, 0.001
current_epoch, max_epochs = 0, 10000

#saving model's weights and biases
saver = tf.train.Saver()


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
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
            print('hypothesis: ', sess.run(hypothesis, feed_dict={x_:nand_inputs, y_:nand_outputs}))
            print('Theta1 Weights: ', sess.run(theta1))
            print('Bias1 : ', sess.run(bias1))
            print('Theta2 Weights: ', sess.run(theta2))
            print('Bias2 : ', sess.run(bias2))


    saver.save(sess, './checkpoints/nand.ckpt', write_meta_graph=False, write_state=False)
    sess.close()

