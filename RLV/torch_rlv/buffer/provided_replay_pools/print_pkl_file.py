import pickle
import pandas as pd
import gym

#just created for testing
dic_rwd_63 = pd.read_pickle('acrobot_avg_rwd_63.pkl', 'gzip')

# Actions
actions_63 = dic_rwd_63['actions']

acr = gym.make('Acrobot-v1')
print(acr.observation_space.shape[0])




