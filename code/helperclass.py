import tensorflow as tf
import tensorflow.keras.losses as kloss
from agents import Agent


class MADDPG:
    #initialize agents for the environments with their parameters leanring rates etc
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.01, fc1=64,
                 fc2=64, gamma=0.99, tau=0.005, chkpt_dir='tmp/maddpg/'):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions


    #main function
    #leanring of actor and critic networks
    def learn(self, memory):
        if not memory.ready():
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_ = memory.sample_buffer()
        

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(states_, dtype=tf.float32)


        all_agents_new_actions = []
        #actions for the new states
        old_agents_actions = []
        #actions that the agent took

        for agent_idx, agent in enumerate(self.agents):
            new_states = tf.convert_to_tensor(actor_new_states[agent_idx], dtype=tf.float32)

            new_pi = agent.target_actor(new_states)

            all_agents_new_actions.append(new_pi)
            
            old_agents_actions.append(actions[agent_idx])
        
        #concat actions of all agents for the critic network
        new_actions = tf.concat(all_agents_new_actions, axis=1)
        old_actions = tf.concat(old_agents_actions, axis=1)

        for agent_idx, agent in enumerate(self.agents):


          with tf.GradientTape() as tape:
            critic_value_ = agent.target_critic(states_, new_actions)
            critic_value_ = tf.reshape(critic_value_, (-1,))
            critic_value = agent.critic(states, old_actions)
            #Q value for current state and actions(which the agent took)
            critic_value = tf.reshape(critic_value, (-1,))
            #critic target Q value = reward(current) + gama*(future rewards) (bellman equation)
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = tf.keras.losses.mean_squared_error(target, critic_value)

    
            critic_grads = tape.gradient(critic_loss, agent.critic.trainable_variables)
            agent.critic.optimizer.apply_gradients(zip(critic_grads, agent.critic.trainable_variables))

        with tf.GradientTape() as tape:
            #actions for the current state predicted by the actor network for both the agents to calculate the Q value for critic
            all_agents_new_mu_actions = []
            for agent_idx in range(2):
              mu_states = tf.convert_to_tensor(actor_states[agent_idx], dtype=tf.float32)
              pi = agent.actor(mu_states)
              all_agents_new_mu_actions.append(pi)
            mu = tf.concat(all_agents_new_mu_actions, axis=1)
            #Gradient Ascent on actor loss or critic value (maximise the reward)
            actor_loss = -agent.critic(states, mu)
            actor_loss = tf.reduce_mean(actor_loss)
            actor_grads = tape.gradient(actor_loss, agent.actor.trainable_variables)
            agent.actor.optimizer.apply_gradients(zip(actor_grads, agent.actor.trainable_variables))
        #update the target networks using soft updates
        agent.update_network_parameters()
