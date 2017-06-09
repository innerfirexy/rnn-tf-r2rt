
# coding: utf-8

# In[37]:


import basic_rnn
import tf_rnn_static
import tf_rnn_dynamic

import tensorflow as tf
import importlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[31]:


"""
RNN_config: Store the configuration information
"""
class RNN_config(object):
    num_steps = 5 # number of truncated backprop steps
    batch_size = 200
    num_classes = 2
    state_size = 4
    learning_rate = 0.1

    def __init__(self, num_steps=5, state_size=4):
        self.num_steps = num_steps
        self.state_size = state_size


# generate data
def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)



def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data and stack them vertically
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i : batch_partition_length * (i+1)]
        data_y[i] = raw_y[batch_partition_length * i : batch_partition_length * (i+1)]
        # Further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps : (i+1) * num_steps]
        y = data_y[:, i * num_steps : (i+1) * num_steps]
        yield (x, y)


def gen_epochs(n, batch_size, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


def train_network(num_epochs, config, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, config.batch_size, config.num_steps)):
            training_loss = 0
            training_state = np.zeros((config.batch_size, config.state_size))
            if verbose:
                print("\nEPOCH", idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ =                     sess.run([losses,
                              total_loss,
                              final_state,
                              train_step],
                                  feed_dict={x:X, y:Y, init_state:training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 100 steps:", training_loss/100)
                    training_losses.append(training_loss/100)
                    training_loss = 0

    return training_losses


def plot_learning_curve(rnn, num_steps, state_size=4, epochs=1, verbose=False):
    global losses, total_loss, final_state, train_step, x, y, init_state
    tf.reset_default_graph()
    g = tf.get_default_graph()

    config = RNN_config(num_steps=num_steps, state_size=state_size)
    losses, total_loss, final_state, train_step, x, y, init_state = rnn.setup_graph(graph=g, config=config)
    res = train_network(epochs, config=config, verbose=verbose)

    plt.plot(res)


# In[32]:


importlib.reload(basic_rnn)
plot_learning_curve(basic_rnn, num_steps=10, state_size=16, epochs=5, verbose=True)


# In[36]:


# importlib.reload(tf_rnn_static)
plot_learning_curve(tf_rnn_static, num_steps=10, state_size=16, epochs=5, verbose=True)


# In[38]:


# importlib.reload(tf_rnn_dynamic)
plot_learning_curve(tf_rnn_dynamic, num_steps=10, state_size=16, epochs=5, verbose=True)


# In[ ]:




