import gym
import torch
from centralized.comnet_tester import CommNet
import pickle

def test_single_env(trial_num):
    env_name = 'Combat-v0'
    #env_name = 'PredatorPrey5x5-v0'
    #random_seed = 543
    #torch.manual_seed(random_seed)
    env = gym.make(env_name)
    #env.seed(random_seed)

    render = False
    gamma = 0.99
    lr = 0.004
    betas = (0.9, 0.999)
    epochs = 30000
    max_iterations = 40
    encoding_size = 50

    num_agents = env.n_agents
    n_actions = env.action_space[0].n
    n_observations = env.observation_space[0].shape[0]

    policy = CommNet(input_features=n_observations,encoding_size=encoding_size, num_actions=n_actions, comm_steps=2)
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=lr)

    running_reward = 0
    losses = []
    rewards = []
    epoch_count = []

    for epoch in range(0, epochs):
        input_state = env.reset()
        iteration = 0
        while iteration < max_iterations:
            action,_,_ = policy(input_state)
            state, reward, done, _ = env.step(action.numpy())
            policy.rewards.append(reward)
            iteration += 1
            if all(done):
                break

        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma) #nagentsx1
        if policy.rewards.mean().sum() < -0.2:
            policy.clearMemory()
            continue
        loss.backward(retain_graph=True)
        optimizer.step()

        running_reward = 0.05 * policy.rewards.mean().sum() + (1 - 0.05) * running_reward
        rewards.append(running_reward)
        losses.append(loss)
        epoch_count.append(epoch)

        policy.clearMemory()
        print('Iteration %s: %f' % (epoch, running_reward))

    with open('commnet_trial_%s.pkl' % trial_num, 'wb') as f:
        pickle.dump({'rewards': rewards, 'losses': losses, 'epochs': epoch_count}, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    test_single_env('combat')
