
# coding: utf-8

# # Simple MNIST convolutional neural network example
# 
# Simple convolutional neural network example

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# In[2]:


learning_rate = 0.001
batch_size = 1000
n_epochs = 20


# In[3]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Display some examples of the training data.

# In[4]:


fig, subplots = plt.subplots(4, 4)

for i in range(16):
    x_index = i % 4
    y_index = i // 4

    ax = subplots[y_index, x_index]
    ax.axis('off')
    ax.imshow(x_train[i], cmap='gray')


# In[5]:


x_batch = tf.placeholder(tf.float32, shape=(None, 28, 28))
y_batch = tf.placeholder(tf.float32, shape=(None, 10))


# Reformat data.

# In[6]:


x_train_input = x_train.astype(np.float32)
y_train_one_hot = tf.one_hot(indices=y_train, depth=10)


# In[7]:


conv_layer = tf.contrib.layers.conv2d(
    inputs=x_batch,
    num_outputs=25,
    kernel_size=3,
    stride=1,
    padding='same',
    activation_fn=None,
    trainable=True
)

flat_layer = tf.layers.flatten(conv_layer)

output_layer = tf.contrib.layers.fully_connected(
    inputs=flat_layer,
    num_outputs=10,
    activation_fn=None,
    trainable=True)

train_loss = tf.losses.sigmoid_cross_entropy(y_batch, output_layer)

adam_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_opt = adam_opt.minimize(loss=train_loss)


# In[8]:


sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)

n_batches = x_train.shape[0] // batch_size

for epoch in range(n_epochs):
    print('epoch:', epoch)
    total_loss = 0.0
    for batch in range(n_batches):
        # Get batch of data
        start_index = batch_size * batch
        end_index = start_index + batch_size
        x_train_batch = x_train[start_index: end_index].astype(np.float32)
        y_train_batch = tf.one_hot(indices=y_train[start_index: end_index], depth=10)
        y_train_batch = y_train_batch.eval(session=sess)
        
        _, loss = sess.run(
            (train_opt, train_loss),
            feed_dict={x_batch: x_train_batch, y_batch: y_train_batch})
        total_loss = total_loss + loss
    print('total_loss:', total_loss)


# Display some examples of the test data.

# In[9]:


fig, subplots = plt.subplots(4, 4)

for i in range(16):
    x_index = i % 4
    y_index = i // 4

    ax = subplots[y_index, x_index]
    ax.axis('off')
    ax.imshow(x_test[i], cmap='gray')


# Use the network to make predictions for the test examples above.

# In[10]:


x_test_batch = x_test[0:16].astype(np.float32)
y_test_batch = tf.one_hot(indices=y_test[0:16], depth=10)
y_test_batch = y_test_batch.eval(session=sess)


predict = tf.argmax(tf.nn.sigmoid(output_layer), axis=1)
        
predicted = sess.run(predict, feed_dict={x_batch: x_test_batch, y_batch: y_test_batch})
print(predicted)


# Not too bad, but definitely room for improvement!
