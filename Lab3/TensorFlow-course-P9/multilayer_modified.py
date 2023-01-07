#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
import time
import random 


batch_size = 8
# Get the number of GPUs available
gpus = tf.Session().list_devices()
num_gpus = sum([1 for gpu in gpus if gpu.device_type == 'GPU'])
print("Num GPUs Available: ", num_gpus)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')

#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
print ( N.shape(data[0][0])[0] )
print ( N.shape(data[0][1])[0] )

train_size = N.shape(data[0][0])[0]
N_GPU = 1

#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[0][1])[0] ):
  real_output[i][data[0][1][i]] = 1.0  


#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0


#set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
y_ = tf.placeholder(tf.float32, [None, 10])



#declare weights and biases
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')




#First convolutional layer: 32 features per each 5x5 patch
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


#Reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height.
#28x28 = 784
#The final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])


#We convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool. 
#The max_pool_2x2 method will reduce the image size to 14x14.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



#Second convolutional layer: 64 features for each 5x5 patch.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#Densely connected layer: Processes the 64 7x7 images with 1024 neurons
#Reshape the tensor from the pooling layer into a batch of vectors, 
#multiply by a weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#drop_out
keep_prob = 0.7
keep_prob = tf.placeholder_with_default(keep_prob, shape=())

#keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Per_image_crossentropy
cross_entropy_local = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

#Crossentropy
cross_entropy = tf.reduce_mean(cross_entropy_local)
#    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# ---------------------------------
#             Cosine decay
# ---------------------------------
batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
  5e-4,                # Base learning rate.
  batch * batch_size,  # Current index into the dataset.
  train_size,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  sess.run(tf.global_variables_initializer())
  #TRAIN 
  print("TRAINING")

  gpus = tf.Session().list_devices()
  num_gpus = sum([1 for gpu in gpus if gpu.device_type == 'GPU'])
  print("Num GPUs Available: ", num_gpus)
  start_time = time.time()

  # -----------------------
  #   Batch size of 50
  # ----------------------
  #num_epochs = 1000
  #for i in range(num_epochs):
    #until 1000 96,35%
   # batch_ini = 50*i
    #batch_end = 50*i+64
  # -----------------------
  #  Batch size of 8
  # ----------------------
  num_epochs = 6250
  # ---------------------  
  # Bathc size of 16
  # --------------------
  #num_epochs = 3125
  #data_train = data[0][0]
  for i in range(num_epochs):
    # ---------------------------------
    # Shuffle the data after each epoch
    # --------------------------------
    #data_list = list(data[0])
    #real_output_list = list(real_output) 
    # Convert data and labels to a list of tuples
    #data_and_labels = list(zip(data_list[0], real_output_list))
    
    # Shuffle the list of tuples
    #random.shuffle(data_and_labels)
    # Unpack the list of tuples back into data and labels
    #data_list[0], real_output_list = zip(*data_and_labels)
    #data[0] = tuple(data_list)
    #real_output = tuple(real_output_list)
    #until 1000 96,35%
    batch_ini =8*i
    batch_end =8*i+8
    
    #for j in range(N_GPU):
    for j in range(num_gpus):
      with tf.device('/gpu:%d' %j):
        batch_xs = data[0][0][batch_ini:batch_end]
        batch_ys = real_output[batch_ini:batch_end]

    if i % 20 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch_xs, y_: batch_ys, keep_prob: 3.0})
      print('step %d, training accuracy %g Batch [%d,%d]' % (i, train_accuracy, batch_ini, batch_end))

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

  print("Training Time: %.3f seconds" % (time.time() - start_time))
  #TEST
  print("TESTING")

  train_accuracy = accuracy.eval(feed_dict={x: data[2][0], y_: real_check, keep_prob: 1.0})
  print('test accuracy %.3f' %(train_accuracy))

  print("Num GPUs Available: ", num_gpus)
