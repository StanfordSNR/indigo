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
        pass
