from uav_env import UAVTASKENV
import random
import numpy as np
from buffer import Buffer
from helperclass import MADDPG
import matplotlib.pyplot as plt


#training of the agents in the environment using MADDPG

#function to concatenate states of each agent for the critic
def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

#creating two UE devices cluster on 600X600 sqmeter area
def create_UE_cluster(x1, y1, x2, y2):
  X = []
  Y = []
  Z = []
  while(len(X)<10):
    cord_x = round(random.uniform(x1,x2),2)
    if(cord_x not in X):
      X.append(cord_x)
  while(len(Y)<10):
    cord_y = round(random.uniform(y1,y2),2)
    if(cord_y not in Y):
      Y.append(cord_y)
  while(len(Z)<10):
      Z.append(0)
  k = []
  i = 0
  while(i<10):
      k.append([X[i],Y[i],Z[i]])
      i += 1
        
  return k

ue_cluster_1 = create_UE_cluster(400, 450, 470, 520)
ue_cluster_2 = create_UE_cluster(30,30,100,100)


#main loop
if __name__ == '__main__':
    
    
    env = UAVTASKENV(ue_cluster_1, ue_cluster_2)
    n_agents = 2
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(3)
    critic_dims = sum(actor_dims)
    PRINT_INTERVAL = 1

    
    n_actions = 22
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=300, fc2=400,  
                           alpha=0.001, beta=0.001,
                           chkpt_dir='tmp/maddpg/')

    memory = Buffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=100)

    
    
    total_steps = 0
    score_history = []
    n_episodes = 100
    timestamp = 200
    avg = []
    

    #standard implemntaion of MADDPG algorithim
    for i in range(n_episodes):
        obs = env.reset()
        score = 0
        episode_step = 0

        for j in range(timestamp):
            
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

           

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 10 == 0:
                maddpg_agents.learn(memory)

            obs = obs_

            score += reward
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
            avg.append(avg_score)

    #plot the final results
    maddpg_agents.save_checkpoint()
    plt.plot(avg)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()