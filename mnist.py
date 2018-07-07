# start date: 06/20/2018

# imports
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import math
from matplotlib import pyplot as plt

# Load data
train = pd.read_csv('Projects/MNIST/train.csv')
print(train.shape)
test = pd.read_csv('Projects/MNIST/test.csv')
print(test.shape)
X_train = (train.ix[:,1:].values).astype('float32')
y_train = (train.ix[:,0].values).astype('int32')
X_test = test.values.astype(np.float)
# print(y_train)
# cv2.imshow('test',X_train[0].reshape(28,28))
# cv2.waitKey(0)

# Preprocess and reshape as 28 x 28 matrices
X_train = X_train.reshape(X_train.shape[0], 28,28,1)
X_test = X_test.reshape(X_test.shape[0], 28,28,1)
# print(X_train.shape)
# print(X_test.shape)

# Standardize, mean center the data
def standardize(x):
    mean_px = x.mean().astype(np.float32)
    std_px = x.std().astype(np.float32)
    return (x-mean_px)/std_px

X_train = standardize(X_train)
X_test = standardize(X_test)

# one hot encoding
indices = (max(y_train) + 1)
total_labels = len(y_train)
y = np.zeros((total_labels,indices))
y[np.arange(total_labels),y_train] = 1
# print(y[0])
# print(y_train[0])


# Splitting data in train-dev-test sets

VALIDATION_SIZE = 2000
Dev_data = X_train[:VALIDATION_SIZE]
Dev_labels = y[:VALIDATION_SIZE]

Train_data = X_train[VALIDATION_SIZE:]
Train_labels = y[VALIDATION_SIZE:]

# print(Dev_data.shape)
# print(Dev_labels.shape)
# print(Train_data.shape)
# print(Train_labels.shape)

# creating input and output placeholders
X = tf.placeholder(tf.float32, shape = [None,784])
y = tf.placeholder(tf.float32, shape = [None, 10])

# W and b initializations
def weight(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# def conv and pool layers
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# Conv layer 1
W_1 = weight([5,5,1,32])
b_1 = bias([32])

x_image = tf.reshape(X, [-1,28,28,1])

h_1 = tf.nn.relu(conv2d(x_image,W_1) + b_1)
h_p1 = max_pool_2x2(h_1)
print(h_p1.shape)

# Conv layer 2
W_2 = weight([5,5,32,64])
b_2 = bias([64])

h_2 = tf.nn.relu(conv2d(h_p1,W_2) + b_2)
h_p2 = max_pool_2x2(h_2)
print(h_p2.shape)

# Conv layer 3
W_3 = weight([5,5,64,128])
b_3 = bias([128])

h_3 = tf.nn.relu(conv2d(h_p2,W_3) + b_3)
h_p3 = max_pool_2x2(h_3)
print(h_p3.shape)

# Dense layers
W_fc1 = weight([4*4*128,784])
b_fc1 = bias([784])
h_p_flat = tf.reshape(h_p3,[-1,4*4*128])

h_fc1 = tf.nn.relu(tf.matmul(h_p_flat,W_fc1) + b_fc1)
print(h_fc1.shape)

# DROPOUT
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer
W_fc2 = weight([784,10])
b_fc2 = bias([10])

y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
print(y_conv.shape)

# mini batches
def batch(X,Y,size,seed = 0):
    np.random.seed(seed)
    m = X.shape[0]
    batches = []

    perm = list(np.random.permutation(m))
    shuffled_X = X[perm,:]
    shuffled_Y = Y[perm,:]

    num_complete_minibatches = math.floor(m/size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[(size*k):(size*(k+1)),:]
        mini_batch_Y = shuffled_Y[(size*k):(size*(k+1)),:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[(size*(k+1)):m, :]
        mini_batch_Y = shuffled_Y[(size*(k+1)):m, :]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        batches.append(mini_batch)

    return batches

# Train and eval

# Parameters
BATCH_SIZE = 512
iter = 50
prob = 0.5

# Learning rate with exponential decay
global_step = tf.Variable(0, trainable=False)
initial_lr = 0.1
Decay_rate = 0.96
Decay_steps = 10
learning_rate = tf.train.exponential_decay(initial_lr, global_step, Decay_steps, Decay_rate, staircase=True)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_conv))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
predict = tf.argmax(y_conv,1)

m = Train_data.shape[0]
seed = 3
losses = []
n_y = Train_labels.shape[1]
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(iter):
    minibatch_loss = 0
    num_batch = int(m / BATCH_SIZE)
    seed+=1
    minibatches = batch(Train_data,Train_labels,BATCH_SIZE,seed)

    for minibatch in minibatches:
        (mini_batch_X,mini_batch_Y) = minibatch

        _, temp_loss = sess.run([optimizer,cost],
            feed_dict = {X:mini_batch_X, y:mini_batch_Y, keep_prob: prob})
        minibatch_loss += temp_loss/num_batches

    losses.append(minibatch_loss)
    train_accuracy = accuracy.eval(feed_dict = {X:mini_batch_X, y:mini_batch_Y, keep_prob:1.0})
    print ('epoch %d, loss %g, training accuracy %g' %(i,loss, train_accuracy))

# Visualize stats
print('dev accuracy %g'%accuracy.eval(feed_dict = {X:Dev_data,y:Dev_labels, keep_prob:1.0}))
plt.plot(np.squeeze(losses))
plt.ylabel('cost')
plt.xlabel('iteration (tens)')
plt.title('loss curve')
plt.show()

# test
y_pred = predict.eval(feed_dict={X: X_test, keep_prob: 1.0})

# Save results
np.savetxt('Projects/MNIST/submission.csv',np.c_[range(1,len(X_test)+1),y_pred],delimiter = ',', header = 'ImageId,Label',comments = '',fmt = '%d')

# end session
sess.close()
