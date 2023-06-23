from networks import ActorNetwork, CriticNetwork
import tensorflow as tf
import numpy

#agent class
#each agent will have two actor and two critic networks
#actor will define the action for a given state
#critic will decide the Q values for that state and action pairs
#target networks for stability in training using soft updates

class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                 alpha=0.01, beta=0.01, fc1=300, fc2=400, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  chkpt_dir=chkpt_dir, name=self.agent_name + '_actor')
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                    chkpt_dir=chkpt_dir, name=self.agent_name + '_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                         chkpt_dir=chkpt_dir, name=self.agent_name + '_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                           chkpt_dir=chkpt_dir, name=self.agent_name + '_target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        #using the actor network given state of the agent return the suitable action currently
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        noise = tf.random.normal(shape=(self.n_actions,))
        action = actions + noise

        return action.numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_weights, weights in zip(self.target_actor.weights, self.actor.weights):
            target_weights.assign(tau * weights + (1 - tau) * target_weights)

        for target_weights, weights in zip(self.target_critic.weights, self.critic.weights):
            target_weights.assign(tau * weights + (1 - tau) * target_weights)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
