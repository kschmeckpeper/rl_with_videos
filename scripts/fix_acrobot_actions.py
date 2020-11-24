import gzip
import pickle
import numpy as np
import argparse

from os.path import join

def combine_replay_pools(args):
    with gzip.open(args.path_1, 'rb') as f:
        data = pickle.load(f)
    
    print("actions:", data['actions'].shape)
    print("before:", data['actions'][:100])
    data['actions'][np.where(data['actions'] > 0.5)] = 1.0
    data['actions'][np.where(data['actions'] < -0.5)] = -1.0
    data['actions'][np.where(np.logical_and(data['actions'] < 0.5, data['actions'] > -0.5))] = 0.0
    print("After:", data['actions'][:100])

    with gzip.open(args.out_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_1', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    combine_replay_pools(args)
