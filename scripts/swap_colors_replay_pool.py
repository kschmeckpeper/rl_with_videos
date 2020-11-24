import gzip
import pickle
import numpy as np
import argparse

from os.path import join

def swap_colors_replay_pools(args):
    with gzip.open(args.path_1, 'rb') as f:
        data_1 = pickle.load(f)
    
    print("data_1", data_1.keys())

    obs = data_1['observations'].reshape(-1, 48, 48, 3)
    obs = obs[:, :, :, ::-1]
    data_1['observations'] = obs.reshape(obs.shape[0], -1)

    with gzip.open(args.paired_out_path, 'wb') as f:
        pickle.dump(data_1, f)
    print("saved paired data to ", args.paired_out_path)


    next_obs = data_1['next_observations'].reshape(-1, 48, 48, 3)
    next_obs = next_obs[:, :, :, ::-1]
    data_1['next_observations'] = next_obs.reshape(next_obs.shape[0], -1)

    print("obs shape:", data_1['observations'].shape)

    with gzip.open(args.out_path, 'wb') as f:
        pickle.dump(data_1, f)
    print("saved flipped data to", args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_1', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('paired_out_path', type=str)
    args = parser.parse_args()

    swap_colors_replay_pools(args)
