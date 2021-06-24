## FetchPush von Gym

# import gym
# env = gym.make('FetchPush-v1')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         observation = observation['observation']
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

import multiworld
import gym

multiworld.register_all_envs()

env = gym.make('SawyerDoorHookResetFreeEnv-v1')

for i_episode in range(60):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        observation = observation['observation']
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()