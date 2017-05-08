import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers, rnn


class ActorCriticNetwork(object):
    def __init__(self, state_dim, action_cnt):
        self.states = tf.placeholder(tf.float32, [None, state_dim])

        actor_h1 = layers.relu(self.states, 8)
        actor_h2 = layers.relu(actor_h1, 8)
        self.action_scores = layers.linear(actor_h2, action_cnt)
        self.action_probs = tf.nn.softmax(self.action_scores)

        critic_h1 = layers.relu(self.states, 8)
        critic_h2 = layers.relu(critic_h1, 8)
        self.state_values = tf.reshape(layers.linear(critic_h2, 1), [-1])

        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


class ActorCriticLSTM(object):
    def __init__(self, state_dim, action_cnt):
        self.states = tf.placeholder(tf.float32, [None, state_dim])
        rnn_in = tf.expand_dims(self.states, [0])

        # create LSTM
        lstm_state_dim = 256
        lstm_cell = rnn.BasicLSTMCell(lstm_state_dim)

        c_init = np.zeros([1, lstm_cell.state_size.c], np.float32)
        h_init = np.zeros([1, lstm_cell.state_size.h], np.float32)
        self.lstm_state_init = (c_init, h_init)

        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.lstm_state_in = (c_in, h_in)

        lstm_outputs, lstm_state_out = tf.nn.dynamic_rnn(
            lstm_cell, rnn_in,
            initial_state=rnn.LSTMStateTuple(c_in, h_in))

        rnn_out = tf.reshape(lstm_outputs, [-1, lstm_state_dim])
        c_out, h_out = lstm_state_out
        self.lstm_state_out = (c_out[:1, :], h_out[:1, :])

        # actor
        self.action_scores = layers.linear(rnn_out, action_cnt)
        self.action_probs = tf.nn.softmax(self.action_scores)

        # critic
        self.state_values = tf.reshape(layers.linear(rnn_out, 1), [-1])

        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
