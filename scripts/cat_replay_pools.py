import gzip
import pickle
import numpy as np
import argparse

from os.path import join

def combine_replay_pools(args):
    with gzip.open(args.path_1, 'rb') as f:
        data_1 = pickle.load(f)
    with gzip.open(args.path_2, 'rb') as f:
        data_2 = pickle.load(f)
    

    combined_data = {}
    for k in data_1.keys():
        combined_data[k] = np.concatenate((data_1[k], data_2[k]), axis=0)
        print(k, data_1[k].shape, data_2[k].shape, combined_data[k].shape)

    with gzip.open(args.out_path, 'wb') as f:
        pickle.dump(combined_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_1', type=str)
    parser.add_argument('path_2', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    combine_replay_pools(args)
