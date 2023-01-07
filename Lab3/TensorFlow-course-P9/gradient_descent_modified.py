import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

lr = [0.1, 0.01, 0.001, 0.0001]

for lr_value in lr:
	# Model parameters
	W = tf.Variable([.3], dtype=tf.float32)
	b = tf.Variable([-.3], dtype=tf.float32)
	# Model input and output
	x = tf.placeholder(tf.float32)
	linear_model = W * x + b
	y = tf.placeholder(tf.float32)

	# loss
	loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
	# optimizer
	optimizer = tf.train.GradientDescentOptimizer(lr_value)
	train = optimizer.minimize(loss)

	# training data
	x_train = [1, 2, 3, 4]
	y_train = [0, -1, -2, -3]
	# training loop
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init) # reset values to wrong

	curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

	loss_list = []
	for i in range(1000):
        	sess.run(train, {x: x_train, y: y_train})
        	curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        	print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
        	loss_list.append(curr_loss)
	print("The size of the loss captured was: ",len(loss_list))
	#Store plots
	# Epochs vs Loss
	plt.plot(loss_list)
	plt.title("Gradient descent optimizer (Loss 0.01)")
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.savefig('gd_loss_vs_epoch.png')	
