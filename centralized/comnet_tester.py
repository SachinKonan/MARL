import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.autograd import Variable

torch.set_default_tensor_type('torch.DoubleTensor')

class CommNet(nn.Module):
    def __init__(self, input_features, encoding_size, num_actions, comm_steps = 1):
        super(CommNet, self).__init__()
        self.encoding_layer = nn.Linear(input_features, encoding_size)
        self.comm_steps = comm_steps

        self.H = nn.Linear(encoding_size, encoding_size)
        self.C = nn.Linear(encoding_size, encoding_size)
        self.H_meta = nn.Linear(encoding_size, encoding_size)
        self.output_layer = nn.Linear(encoding_size, num_actions)
        self.temp_value_layer = nn.Linear(encoding_size, encoding_size)
        self.value_layer = nn.Linear(encoding_size, 1)

        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.entropy = []
        
        self.eps = 1e-7

    def forward(self, input_data):
        input_data = torch.Tensor(input_data)
        """if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(0)"""
        n_agents, feature_dim = input_data.shape
        input_data = self.encoding_layer(input_data)

        h_i = input_data
        c_i = torch.zeros(size=(input_data.shape), requires_grad=True)
        h_0 = input_data.clone()

        for _ in range(self.comm_steps):
            h_instant = self.H(h_i)
            h_skip_instant = self.H_meta(h_0)
            c_instant = self.C(c_i)

            #sum_for_all = [sum_x1 + sum_x2 + sum_x3....sum_x_n_observations]
            h_i = torch.tanh(h_instant + h_skip_instant + c_instant) #nagentsxencoding_size
            sum_for_all = h_i.sum(dim=0).unsqueeze(0).repeat(n_agents,1) #sum of h_i, repeated for n_agents
            c_i = (sum_for_all - h_i)*1/(n_agents-1)

        epsilon = 1e-5

        state_value = self.value_layer(self.temp_value_layer(h_i))
        action_probs = F.softmax(self.output_layer(h_i), dim=-1)
        action_distribution = Categorical(action_probs)

        action = action_distribution.sample()
        logprob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        self.logprobs.append(logprob)
        self.state_values.append(state_value)
        self.entropy.append(entropy)
        return action, state_value, logprob

    def calculateLoss(self, gamma=0.99):
        # calculating discounted rewards:
        rewards = np.array(self.rewards, dtype=np.float64)
        iterations, num_agents = rewards.shape
        start = np.zeros((num_agents))
        for i in range(iterations-1, -1, -1):
            rewards[i] += start*gamma
            start = rewards[i]
        self.rewards = rewards
        # normalizing the rewards:
        rewards = torch.from_numpy(rewards)
        #rewards = (rewards - rewards.mean(0)) / (rewards.std(0) + self.eps)
        loss = torch.zeros((iterations, num_agents, 1))
        i = 0
        for logprob, value, reward, entropy in zip(self.logprobs, self.state_values, rewards, self.entropy):
            logprob = logprob.view(num_agents, 1)
            value = value.view(num_agents, 1)
            reward = reward.view(num_agents, 1)
            entropy = entropy.view(num_agents,1)
            
            advantage = reward - value
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            loss[i] = action_loss + value_loss + entropy
            #loss[i] = action_loss + value_loss
            i+=1

        #average loss per agent across timesteps:
        meaned_loss = loss.mean(dim=0)
        #total loss across agents:
        summed_across_agents = meaned_loss.sum()
        return summed_across_agents

    def clearMemory(self):
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.entropy = []

if __name__ == '__main__':
    x = torch.randn((16, 4,28))
    policy = CommNet(28, 50, 5, comm_steps=2)
    actions, s_value, probs = policy(x)
    print(actions.shape)
    print(policy.rewards)