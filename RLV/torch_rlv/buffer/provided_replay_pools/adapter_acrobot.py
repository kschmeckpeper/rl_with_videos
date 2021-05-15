import pandas as pd
import numpy as np
import os
import torch as T


def get_acrobot_observations_actions():
    current_directory = os.path.dirname(__file__)

    # Unpickle
    dic_rwd_63 = pd.read_pickle(os.path.join(current_directory, 'acrobot_avg_rwd_63.pkl'), 'gzip')
    dic_rwd_79 = pd.read_pickle(os.path.join(current_directory, 'acrobot_avg_rwd_79.pkl'), 'gzip')
    dic_rwd_99 = pd.read_pickle(os.path.join(current_directory, 'acrobot_avg_rwd_99.pkl'), 'gzip')

    # Obtain Observations - Inputs
    obs = (dic_rwd_63['observations'], dic_rwd_79['observations'], dic_rwd_99['observations'])
    observations = np.concatenate(obs)

    # Obtain Actions - Output
    acs = (dic_rwd_63['actions'], dic_rwd_79['actions'], dic_rwd_99['actions'])
    actions = np.concatenate(acs)

    return T.from_numpy(observations), T.from_numpy(actions)




