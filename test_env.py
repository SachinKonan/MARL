import gym
import ma_gym

env = gym.make('PredatorPrey5x5-v0')
done_n = [False for _ in range(env.n_agents)]
ep_reward = 0

obs_n = env.reset()
while not all(done_n):
    env.render()
    x, y = input("Enter a two value: ").split()
    obs_n, reward_n, done_n, info = env.step([int(x),int(y)])
    ep_reward += sum(reward_n)
    print(ep_reward)
env.close()