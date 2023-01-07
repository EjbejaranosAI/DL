import tensorflow as tf
import read_inputs
import numpy as N
import matplotlib.pyplot as plt

#Variables
lr = 0.5



def train_dataset(learning_rate):
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
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


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
  return train_accs, train_losses, valid_accs, valid_losses



def plot_acc_loss(acc, loss, valid_acc, valid_loss, save_fig=True, fig_name='acc_loss.png'):
  # Plot training accuracy and loss
  plt.subplot(2, 1, 1)
  plt.plot(acc)
  plt.plot(valid_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='lower right')

  plt.subplot(2, 1, 2)
  plt.plot(loss)
  plt.plot(valid_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper right')
  plt.tight_layout()

  if save_fig:
    plt.savefig(fig_name)
  else:
    plt.show()

train_acc, train_loss, valid_acc, valid_loss = train_dataset(lr)
plot_acc_loss(train_acc, train_loss, valid_acc, valid_loss)
