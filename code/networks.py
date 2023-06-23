import tensorflow as tf
import os

#critic Network
class CriticNetwork(tf.keras.Model):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.q = tf.keras.layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=beta)
        

    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.q(x)

        return q

    def save_checkpoint(self):
        self.save_weights(self.chkpt_file)

    def load_checkpoint(self):
        self.load_weights(self.chkpt_file)

#actor network
class ActorNetwork(tf.keras.Model):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.pi = tf.keras.layers.Dense(n_actions, activation='tanh')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        pi = self.pi(x)

        return pi

    def save_checkpoint(self):
        self.save_weights(self.chkpt_file)

    def load_checkpoint(self):
        self.load_weights(self.chkpt_file)
