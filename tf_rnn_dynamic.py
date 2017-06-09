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
        Inputs
        """
        rnn_inputs = tf.one_hot(x, config.num_classes)

        """
        RNN
        """
        cell = tf.contrib.rnn.BasicRNNCell(config.state_size)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

        """
        Predictions, loss, training step
        """
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [config.state_size, config.num_classes])
            b = tf.get_variable('b', [config.num_classes], initializer=tf.constant_initializer(0.0))
        logits = tf.reshape(
                    tf.matmul(tf.reshape(rnn_outputs, [-1, config.state_size]), W) + b,
                    [config.batch_size, config.num_steps, config.num_classes])
        predictions = tf.nn.softmax(logits)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        total_loss = tf.reduce_mean(losses)
        train_step = tf.train.AdagradOptimizer(config.learning_rate).minimize(total_loss)

        return losses, total_loss, final_state, train_step, x, y, init_state
