#!/usr/bin/env python
import tensorflow as tf
#Store plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
lr_value = 0.01

# Momentum
#name_optimizer = 'Momentum'
#optimizer = tf.train.MomentumOptimizer(lr_value, momentum=0.9)
# SGD
#name_optimizer = 'GradientDescentOptimizer'
#optimizer = tf.train.GradientDescentOptimizer(lr_value)
# Nesterov Accelerated Gradient (NAG)
#name_optimizer = 'NesterovAccelerated(NAG)'
#optimizer = tf.train.MomentumOptimizer(lr_value, momentum=0.9,use_nesterov=True)
# Adagradd
#name_optimizer = 'Adagrad'
#optimizer = tf.train.AdagradOptimizer(lr_value)
# Adadelta
name_optimizer = 'Adadelta'
optimizer = tf.train.AdadeltaOptimizer(lr_value)
# Adam
#name_optimizer = 'Adam'
#optimizer = tf.train.AdamOptimizer(lr_value)
# RMSProp
#name_optimizer = 'RMSprop'
#optimizer = tf.train.RMSPropOptimizer(lr_value)
# Ftrl optimizer
#name_optimizer = 'Ftrl'
#optimizer = tf.train.FtrlOptimizer(learning_rate=0.01, l1_regularization_strength=0.001, l2_regularization_strength=0.001)
# Proximal adagrad optimizer
#name_optimizer = 'ProximalAdagrad'
#optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.001, l1_regularization_strength=0.0001)
# Proximal gradient Descent optimizer
#name_optimizer = 'ProximalGradientDescent'
#optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=0.0001, l1_regularization_strength=0.00001)

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

# Logaritmic scale
plt.figure()
plt.semilogy(loss_list)
plt.title("Logratitmic scale-"+name_optimizer+" optimizer (Lr = {})".format(lr_value))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("Log_"+name_optimizer+"_optimizer(Lr_{})".format(lr_value)+'.png')


# Epochs vs Loss
plt.figure()
plt.plot(loss_list)
plt.title(name_optimizer+" optimizer(noLimit) (Lr = {})".format(lr_value))
plt.xlabel('epoch')
plt.ylabel('loss')
#plt.ylim(0, 20)
plt.savefig(name_optimizer+"_NoLimit__optimizer(Lr_{})".format(lr_value)+'.png')
