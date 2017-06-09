import tensorflow as tf
import numpy as np


"""
setup_graph
"""
def setup_graph(graph, config):
    with graph.as_default():
        """
        Placeholders
        """
        x = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name='input_placeholder')
        y = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name='labels_placeholder')
        init_state = tf.zeros([config.batch_size, config.state_size])

        """
        Inputs and labels
        """
        x_one_hot = tf.one_hot(x, config.num_classes)
        rnn_inputs = tf.unstack(x_one_hot, axis=1)
        y_as_list = tf.unstack(y, num=config.num_steps, axis=1)

        """
        RNN
        """
        cell = tf.contrib.rnn.BasicRNNCell(config.state_size)
        rnn_outputs, final_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, initial_state=init_state)

        """
        Predictions, loss, training step
        """
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [config.state_size, config.num_classes])
            b = tf.get_variable('b', [config.num_classes], initializer=tf.constant_initializer(0.0))
        logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
        predictions = [tf.nn.softmax(logit) for logit in logits]

        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for \
                  logit, label in zip(logits, y_as_list)]
        total_loss = tf.reduce_mean(losses)
        train_step = tf.train.AdagradOptimizer(config.learning_rate).minimize(total_loss)

        return losses, total_loss, final_state, train_step, x, y, init_state
