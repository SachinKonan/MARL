import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
import ma_gym

"""
Action Meanings:
0: 'DOWN'
1: 'LEFT'
2: 'UP'
3: 'RIGHT'
4: 'NOOP'
5: 'Attack Opponent 0'
6: 'Attack Opponent 1'
7: 'Attack Opponent 2'
8: 'Attack Opponent 3'
9: 'Attack Opponent 4'

env.get_action_meanings() -> returns action definitions for each agent
env._grid_shape -> returns size of the grid
observation (n_agentsx10*n_agents = 5x150): {
    [0:n_agents(5)]: one_hot_encoding of id,
    [5: 7]: normalized location (x,y),
    [7:8]: agent health,
    [8:9]: agent_cool
}
"""

class Actor_Critic(nn.Module):
    def __init__(self, n_actions, n_observations):
        super().__init__()
        self.affine1 = nn.Linear(n_observations, 128)

        self.output_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

        self.eps = 1e-09
        self.reset_memory()

    def reset_memory(self):
        self.log_probs = []
        self.rewards = []
        self.state_values = []

    def forward(self, input_obs):
        x = F.relu(self.affine1(input_obs))
        action_prob_dist = F.softmax(self.output_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_prob_dist, state_value

    def add_memory(self, log_prob, reward, state_value):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.state_values.append(state_value)

    def convert_rewards(self, gamma=0.99):
        rewards = np.array(self.rewards, dtype=np.float64)
        n = rewards.shape[0]
        start = 0
        for i in range(n-1, -1, -1):
            rewards[i] += start*gamma
            start = rewards[i]
        rewards = np.array(rewards)
        #rewards = (rewards - rewards.mean(axis=0)) / (rewards.std(axis=0) + self.eps)
        self.rewards = rewards

def construct_adj_matrix(env):
    agent_positions = env.agent_pos
    h, w = env._grid_shape
    MAX_DISTANCE = np.sqrt(h**2 + w**2)
    adj_mat = np.zeros((env.n_agents, env.n_agents))
    for agent, pos in agent_positions.items():
        for other_agent, other_pos in agent_positions.items():
            adj_mat[agent][other_agent] = np.sqrt((pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2)
            adj_mat[agent][other_agent] /= MAX_DISTANCE
    return adj_mat

def neighborhood(i, env, comm_range=0.1):
    adj_mat = construct_adj_matrix(env)
    neighbors = []
    for other_agent in range(env.n_agents):
        if other_agent != i and adj_mat[i][other_agent] < comm_range:
            neighbors.append(other_agent)
    return neighbors

def main():
    random_seed = 543
    torch.manual_seed(random_seed)
    env = gym.make('Combat-v0')
    env.seed(random_seed)

    epochs = 1000
    max_iterations = 200
    gamma = 0.95
    lr = 3e-2
    lr_decay = 0.99

    num_agents = env.n_agents
    n_actions = env.action_space[0].n
    n_observations = env.observation_space[0].shape[0]
    obs = env.reset()
    agents = [Actor_Critic(n_actions, n_observations) for _ in range(num_agents)]
    optimizers = [optim.Adam(agents[agent].parameters(), lr) for agent in range(num_agents)]

    epoch_data = []
    global_rewards = []
    running_reward = 0

    for epoch in range(epochs):
        iteration = 0
        for agent in agents:
            agent.reset_memory()
        input_state = env.reset()
        done = [False for i in range(num_agents)]
        while iteration < max_iterations and all(done) == False:
            #actor step
            info = []
            for i in range(num_agents):
                input_obs = input_state[i] # obs for the ith agent
                probs, state_val = agents[i].forward(torch.Tensor(input_obs))
                # probs: [prob(action) for action in range(action_space)] state_val: V(S)
                distribution = Categorical(probs)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
                action = action.item()
                info.append([action, log_prob, state_val])

            input_state, rewards, done, _ = env.step([agent[0] for agent in info])
            for i in range(0, len(info)):
                log_prob = info[i][1]
                state_val = info[i][2]
                agents[i].add_memory(log_prob, rewards[i], state_val)

            #consensus update
            for i in range(num_agents):
                neighbors = neighborhood(i, env)
                num_neighbors = len(neighbors)
                for j in neighbors:
                    agents[i].value_head.weight.data += agents[j].value_head.weight.data
                agents[i].value_head.weight.data /= (num_neighbors + 1)
            iteration+=1

        #after iterations
        #expected long term rewards
        for i in range(num_agents):
            agents[i].convert_rewards(gamma)

        losses = [[None for j in range(num_agents)] for i in range(iteration)]

        for i in range(iteration):
            for agent in range(0, num_agents):
                advantage = agents[agent].rewards[i] - agents[agent].state_values[i]
                #advantage = agents[agent].rewards[i] - advantage
                losses[i][agent] = agents[agent].log_probs[i]*advantage + torch.abs(agents[agent].state_values[i] - agents[agent].rewards[i]) # actor_loss + critic_loss
        losses_per_agent = [torch.stack([losses[i][agent] for i in range(iteration)]).mean() for agent in range(num_agents)]

        for agent in range(num_agents):
            optimizers[agent].zero_grad()
            losses_per_agent[agent].backward()
            optimizers[agent].step()

        mean_reward_per_agents = np.array([agent.rewards.mean() for agent in agents])
        running_reward = 0.05 * mean_reward_per_agents.mean() + (1 - 0.05) * running_reward
        global_rewards.append(running_reward)
        epoch_data.append(abs(torch.Tensor(losses_per_agent).mean().item()))

        print('Epoch %s Global Loss: %s' % (epoch, epoch_data[-1]))
        lr *= lr_decay

    np.save('rewards_decentral', np.array(global_rewards))
    np.save('grad_loss_decentral', np.array(epoch_data))

main()