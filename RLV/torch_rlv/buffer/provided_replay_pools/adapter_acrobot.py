import pandas as pd
import numpy as np
import os
import torch as T


def get_acrobot_data(index_from, index_to):
    current_directory = os.path.dirname(__file__)

    # Unpickle
    dic_rwd_63 = pd.read_pickle(os.path.join(current_directory, 'acrobot_avg_rwd_63.pkl'), 'gzip')
    dic_rwd_79 = pd.read_pickle(os.path.join(current_directory, 'acrobot_avg_rwd_79.pkl'), 'gzip')
    dic_rwd_99 = pd.read_pickle(os.path.join(current_directory, 'acrobot_avg_rwd_99.pkl'), 'gzip')

    # Obtain Observations - Inputs
    obs = (dic_rwd_63['observations'], dic_rwd_79['observations'], dic_rwd_99['observations'])
    observations = np.concatenate(obs)

    # Obtain Next States
    nxt = (dic_rwd_63['next_observations'], dic_rwd_79['next_observations'], dic_rwd_99['next_observations'])
    next_observations = np.concatenate(nxt)

    # Obtain Terminal Boolean
    term = (dic_rwd_63['terminals'], dic_rwd_79['terminals'], dic_rwd_99['terminals'])
    terminal_states = np.concatenate(term)

    # Obtain Actions - Targets for RLV
    tar = (dic_rwd_63['actions'], dic_rwd_79['actions'], dic_rwd_99['actions'])
    target = np.concatenate(tar)

    return T.from_numpy(observations[index_from:index_to]), T.from_numpy(next_observations[index_from:index_to]), \
           T.from_numpy(terminal_states[index_from:index_to]), T.from_numpy(target[index_from:index_to])
