import tensorflow as tf
import numpy as np

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

train_labels = onehot(train_y_1)
eval_labels = onehot(test_y_1)

train_x_3 = padd(train_x_2)
test_x_3 = padd(test_x_2)

train_data = matt(train_x_3)
eval_data = matt(test_x_3)
    
train_x = np.array(train_data)
test_x = np.array(eval_data)
train_y = np.array(train_labels)
test_y = np.array(eval_labels)



print (len(train_y))
print (len(train_y[0]))





# declairing the no of classes and batch size
n_classes = 2
batch_size = 10

# declairing x and y two tensorflow functions
# the x variable will be given a 28*28 matrix
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


# for dropout layer
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

# creates a convolutional 2D layer
def conv2d(x, W):
    # defined in middle elements  stride 1*1
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# maxpooling on convolution layer
def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# defining the network
def convolutional_neural_network(x):

    # [5,5,1,32] => filter of 5X5, take one input and produce 32 output features
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    # new shape is 28*28*1 and -1 does the flatening of the layer
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


# training the model
def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    #cost = tf.reduce_mean( tf.nn.softmax(logits=prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0




            # running for the whole training data in batchs
            i = 0
            while i < len(train_x)-1:
                start = i
                end = i + batch_size

                epoch_x = np.array(train_x[start:end])
                epoch_y = np.array(train_y[start:end])
                             
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

                i = i+1





                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',accuracy.eval({x:test_x, y:test_y})*100.0, "%")
            print ('---------------------------------------------')

            
train_neural_network(x)








