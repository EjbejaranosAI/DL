import tensorflow as tf
import read_inputs
import numpy as N
import matplotlib.pyplot as plt

def train_dataset(lr_value):
  # read data from file
  data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
  data = data_input[0]


  # data layout changes since output should an array of 10 with probabilities
  real_output = N.zeros((N.shape(data[0][1])[0], 10), dtype=N.float)
  for i in range(N.shape(data[0][1])[0]):
    real_output[i][data[0][1][i]] = 1.0

  # data layout changes since output should an array of 10 with probabilities
  real_check = N.zeros((N.shape(data[2][1])[0], 10), dtype=N.float)
  for i in range(N.shape(data[2][1])[0]):
    real_check[i][data[2][1][i]] = 1.0

  # set up the computation. Definition of the variables.
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  y_ = tf.placeholder(tf.float32, [None, 10])
  
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  #------------- 
  # OPTIMIZERS
  #------------
  # Gradient descent
  #name_opt = "Gradient Descent Optimizer"
  #optimizer = tf.train.GradientDescentOptimizer(lr_value)
 
  # Momentum
  #name_opt = 'Momentum'
  #optimizer = tf.train.MomentumOptimizer(lr_value, momentum=0.9)
  
  # Adam
  #name_opt = 'Adam'
  #optimizer = tf.train.AdamOptimizer(lr_value)

  # Nesterov Accelerated Gradient (NAG)
  #name_opt = 'Nesterov Accelerated (NAG)'
  #optimizer = tf.train.MomentumOptimizer(lr_value, momentum=0.9,use_nesterov=True)

  # Proximal gradient Descent optimizer
  name_opt = 'ProximalGradientDescent'
  optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=lr_value, l1_regularization_strength=0.01)
  
  #------------
  # TRAIN STEP
  #------------  
  train_step = optimizer.minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Lists to store the training and validation accuracy and loss by epoch
  train_accs = []
  train_losses = []
  valid_accs = []
  valid_losses = []
  # Define the accuracy and cross entropy tensors
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # TRAINING PHASE
  print("TRAINING")

  for i in range(500):
    batch_xs = data[0][0][100 * i:100 * i + 100]
    batch_ys = real_output[100 * i:100 * i + 100]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Compute training accuracy and loss
    train_acc, train_loss = sess.run([accuracy, cross_entropy], feed_dict={x: data[0][0], y_: real_output})
    # Compute validation accuracy and loss
    valid_acc, valid_loss = sess.run([accuracy, cross_entropy], feed_dict={x: data[1][0], y_: real_check[:N.shape(data[1][1])[0]]})

    # Append current epoch's training and validation accuracy and loss to respective lists
    train_accs.append(train_acc)
    train_losses.append(train_loss)
    valid_accs.append(valid_acc)
    valid_losses.append(valid_loss)



  #CHECKING THE ERROR
  print("ERROR CHECK")



  print(f"Train accuracy: {train_acc}")
  print(f"Train loss: {train_loss}")
  print(f"Validation accuracy: {valid_acc}")
  print(f"Validation loss: {valid_loss}")
  return train_accs, train_losses, valid_accs, valid_losses, name_opt



def plot_acc_loss(acc, loss, valid_acc, valid_loss, lr, name_opt ,save_fig=True):

  fig_name="acc_loss(Lr = {})".format(lr)+"_"+name_opt+".png"
  #plt.figure(figsize=(20, 10))
  # Plot training accuracy and loss
  plt.subplot(2, 1, 1)
  plt.plot(acc)
  #plt.plot(valid_acc)
  plt.title("Accuracy vs Epochs (Lr = {})".format(lr)+""+name_opt)
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  #plt.legend(["-LR:"+str(lr)], loc='lower right')
  plt.text(len(acc)-1, acc[-1], "----->Acc:  {:.2f}".format(acc[-1])+"-LR:"+str(lr))

  plt.subplot(2, 1, 2)
  plt.plot(loss)
  #plt.plot(valid_loss)
  plt.title("Loss vs Epochs (Lr = {})".format(lr)+""+name_opt)
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  #plt.legend(["-LR:"+str(lr)], loc='upper right')
  plt.text(len(loss)-1, loss[-1], "----->Loss:  {:.2f}".format(loss[-1])+"LR: "+str(lr))
  plt.tight_layout()

  if save_fig:
    plt.savefig(fig_name)
  else:
    plt.show()

#------------------
# Array with LR
#-----------------
lr_array = [0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
plt.figure(figsize=(22,16))
#------------------
#  Magic is iterate
#------------------
for lr in lr_array:

  train_acc, train_loss, valid_acc, valid_loss,opt = train_dataset(lr)
  plot_acc_loss(train_acc, train_loss, valid_acc, valid_loss,lr=lr,name_opt=opt)



