import numpy as np
import tensorflow as tf


class ActorNetwork(object):
    def __init__(self, state_dim, action_cnt):
        self.state_dim = state_dim
        self.action_cnt = action_cnt

    def build_actor_network(self, state):
        h1_dim = 20
        W1 = tf.get_variable('W1', [self.state_dim, h1_dim],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [h1_dim],
                             initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.tanh(tf.matmul(state, W1) + b1)

        W2 = tf.get_variable('W2', [h1_dim, self.action_cnt],
                             initializer=tf.random_normal_initializer())
        b2 = tf.get_variable('b2', [self.action_cnt],
                             initializer=tf.constant_initializer(0.0))

        action_scores = tf.matmul(h1, W2) + b2
        return action_scores


class CriticNetwork(object):
    def __init__(self, state_dim):
        self.state_dim = state_dim

    def build_critic_network(self, state):
        h1_dim = 20
        W1 = tf.get_variable('W1', [self.state_dim, h1_dim],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [h1_dim],
                             initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.tanh(tf.matmul(state, W1) + b1)

        W2 = tf.get_variable('W2', [h1_dim, 1],
                             initializer=tf.random_normal_initializer())
        b2 = tf.get_variable('b2', [1],
                             initializer=tf.constant_initializer(0.0))

        state_value = tf.matmul(h1, W2) + b2
        return state_value
