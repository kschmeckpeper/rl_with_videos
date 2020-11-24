import gzip
import pickle
import numpy as np
import argparse

import matplotlib.pyplot as plt

from os.path import join, isdir
from os import listdir



def get_reward_stats(path, out_path, name):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)

    rewards = [0.0]
    steps = [0]

    for i in range(data['rewards'].shape[0]):
        rewards[-1] += data['rewards'][i]
        steps[-1] += 1
        if data['terminals'][i]:
            rewards.append(0.0)
            steps.append(0)


    if steps[-1] == 0:
        steps = steps[:-1]
        rewards = rewards[:-1]
    rewards = np.array(rewards)

    if out_path is not None:
        plt.hist(rewards)
        plt.savefig(out_path)
        plt.clf()

    #print("rewards:", rewards)
    #print("num steps:", steps)

    print("\n")
    print(name)
    print("num trajectories:", rewards.shape[0])
    print("max reward:", np.max(rewards))
    print("min reward:", np.min(rewards))

    print("Average reward", np.mean(rewards))
    print("std dev:", np.std(rewards))
    print("std error", np.std(rewards) / np.sqrt(rewards.shape[0]))
    return np.mean(rewards)

def get_all_reward_stats(path, out_path):
    dirs = [d for d in listdir(path) if isdir(join(path, d)) and "checkpoint" in d]
    dirs = sorted(dirs)

    num_steps = []
    rewards = []
    for d in dirs:
        rewards.append(get_reward_stats(join(path, d, "replay_pool.pkl"), out_path+"_"+d+".png", d))
        num_steps.append(int(d.split('_')[1]))

    num_steps, rewards = (list(t) for t in zip(*sorted(zip(num_steps, rewards))))

    plt.plot(num_steps, rewards)
    plt.savefig(out_path+"_all_checkpoints.png")
    plt.clf()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()

    if args.path[-4:] == ".pkl":
        get_reward_stats(args.path, args.out_path, "Replay pool")
    else:
        get_all_reward_stats(args.path, args.out_path)
