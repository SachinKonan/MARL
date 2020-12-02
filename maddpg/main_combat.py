from maddpg.MADDPG import MADDPG
import numpy as np
import torch as th
from maddpg.params import scale_reward
import gym
import ma_gym

# do not render the scene

env_name = 'Combat-v0'
#random_seed = 543
#torch.manual_seed(random_seed)
env = gym.make(env_name)

reward_record = []

np.random.seed(1234)
th.manual_seed(1234)

n_agents = env.n_agents
n_actions = env.action_space[0].n
n_states = env.observation_space[0].shape[0]

capacity = 1000000
batch_size = 1000

n_episode = 30000
max_steps = 40
episodes_before_train = 100

win = None
param = None

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = env.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        obs_, reward, done, _ = env.step(action.numpy())

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs
        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

np.save('rewards_combat', reward_record)