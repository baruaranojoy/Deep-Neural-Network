from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(7)





def avg(x):
    sun = 0.0
    count = 0.0
    for k in range(len(x)):
        data = x[k]
        for m in range(len(data)):
            sun = sun + data[m]
            count = count + 1
    return sun/count


def convert(train_x_1, average):
    data = []
    for k in range(len(train_x_1)):
        da = train_x_1[k]
        one = []
        for m in range(len(da)):
            if da[m] > average:
                one.append(1)
            else:
                one.append(0)
        data.append(one)
    return data

def onehot(train_y_1):
    data = []
    for k in range(len(train_y_1)):
        da = train_y_1[k]
        if da == 0.:
            data.append([1,0])
        else:
            data.append([0,1])
    return data

def padd(train_x_2):
    data = []
    for k in range(len(train_x_2)):
        da = train_x_2[k]
        prio = []
        for m in range(10):
            prio.append(0)
        for n in range(len(da)):
            prio.append(da[n])
        for p in range(10):
            prio.append(0)
        data.append(prio)
    return data

def matt(train_x_3):
    data = []
    for k in range(len(train_x_3)):
        da = train_x_3[k]
        pad = []
        pad.append(da)
        for m in range(len(da)-1):
            pad.append([0]*len(da))
        data.append(pad)

    data1 = []
    for k in range(len(data)):
        dd = data[k]
        sent = []
        for m in range(len(dd)):
            ds = dd[m]
            for n in range(len(ds)):
                sent.append(float(ds[n]))
        data1.append(sent)
        
    return data1


dataset = np.loadtxt("data\data.csv", delimiter=",")

x = dataset[:,0:8] 
y = dataset[:,8]

size = int(len(x)*0.7)

train_x_1 = x[:size]
train_y_1 = y[:size]

test_x_1 = x[size:]
test_y_1 = y[size:]
# reading the data complete


average = avg(x)
#train_x_2 = convert(train_x_1, average)
#test_x_2 = convert(test_x_1, average)

train_x_2 = train_x_1
test_x_2 = test_x_1

train_labels = train_y_1
eval_labels = test_y_1

train_x_3 = padd(train_x_2)
test_x_3 = padd(test_x_2)

train_data = matt(train_x_3)
eval_data = matt(test_x_3)
    
train_data = np.array(train_data)
eval_data = np.array(eval_data)
train_labels = np.array(train_labels)
eval_labels = np.array(eval_labels)


learning_rate = 0.01
training_epochs = 1000
batch_size = 128
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=decoder_op, labels=train_data))
#optimizer = tf.train.AdamOptimizer().minimize(cost)
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    total_batch = int(len(train_data)/batch_size)
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            start = i
            end = i + batch_size
            batch_x = np.array(train_data[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})

        print("Epoch:", '%04d' % (epoch+1),"cost =", "{:.9f}".format(c))
        print ('---------------------------------------------')

    print("Optimization Finished!")

    
