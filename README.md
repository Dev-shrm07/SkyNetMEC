
# SkyNetMEC

This repository contains an implementation of a UAV-based MEC model using the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) reinforcement learning algorithm for task offloading and trajectory planning in the UAVs. The model is designed to optimize task execution by multiple UAVs in a custom environment consisting of multiple UE clusters and fog devices. In this model it is stimulated for 2 UAV devices with 2 UE clusters and 2 fog devices.

## Environment Description

The custom environment simulates a scenario where two UAVs interact with 2 UE clusters and 2 fog devices. The UAV devices receive task requests from the UE clusters. The UAVs make decisions on the fraction of the task to be executed locally and the fraction to be offloaded to the fog devices. Additionally, the model decides the movement distance and angle for the UAVs to improve coverage of the UE devices. The Reward for this model is the negative of the system cost which is the sum of total time, energy and reciprocal of throughput (as we have to maximize it) during the execution of all the tasks in a timestamp.

The calculations for Datarate, time and energy consumptions have been referenced from different research papers.

## Problem Description

The task offloading problem involves two UAV agents, each equipped with one actor network, one critic network, one target actor network, and one target critic network. The goal is to determine the fraction of a task that should be executed locally on the UAV and the fraction that should be offloaded to a fog device. Additionally, the UAV agents need to decide their movement parameters, including distance and angle, to ensure better coverage of the UE (User Equipment) clusters.

It can be thought of as Makrov's Decision problem
- State: 3D coordinates of the UAV
- Action space (for each UAV): distance, angle from the horizontal scale, task splitting ratios for each UE device

Reward: Negative of system cost as discussed, applied with a penalty of coverage if calculating that

## Algorithm Overview

The MADDPG algorithm enables the UAV agents to learn optimal task offloading policies through a process of exploration and learning. Here is an overview of the algorithm:

1. Network Initialization: The actor and critic networks for each UAV agent were initialized, along with their corresponding target networks. The required hyperparameters such as learning rate, discount factor (gamma), and soft update rate (tau) were set.

2. Replay Buffer Initialization: A replay buffer was created to store the experiences of the agents. The replay buffer facilitated experience replay, breaking temporal correlations in the training data.

3. Agent Action Selection: Each UAV agent selected an action based on its current observation using its actor network. The actor networks determined the fraction of the task to be executed locally and the fraction to be offloaded to the fog.

4. Environment Interaction: The selected actions were executed in the environment, and the resulting next states, rewards, and task completion indicators (dones) were observed.

5. Experience Storage: The experiences (states, actions, rewards, next states, and dones) were stored in the replay buffer for each UAV agent.

6. Training Loop:
   - A mini-batch of experiences was sampled from the replay buffer.
   - The critic networks were updated by computing target Q-values and minimizing the critic loss using gradient descent.
   - The actor networks were updated by maximizing the actor loss using gradient ascent.
   - A soft update of the target networks was performed to track the main networks.
   - Repeated Steps 3-6: The process of action selection, environment interaction, experience storage, and training was repeated for a specified number of iterations or until convergence.

7. Evaluation and Deployment: The Model can be evaluated using the `test.py` file in the code folder by loading the weights of trained actor-critic networks.

## References

Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, O. P., & Mordatch, I. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems (pp. 6382-6393).

Z. Yu, Y. Gong, S. Gong, and Y. Guo, "Joint Task Offloading and Resource Allocation in UAV-Enabled Mobile Edge Computing," IEEE Transactions on Vehicular Technology, vol. 68, no. 4, pp. 3114-3127, April 2019.
