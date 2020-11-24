import pickle
import gzip
import argparse

import cv2
import numpy as np
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('pool', type=str)
parser.add_argument('out_path', type=str)
parser.add_argument('--size', type=int, default=128)
args = parser.parse_args()

with gzip.open(args.pool, 'rb') as f:
    data = pickle.load(f)

print("Keys:", data.keys())

for i in tqdm.tqdm(range(data['observations'].shape[0])):
    obs = data['observations'][i]

    obs = obs.reshape(48, 48, 3)


    next_obs = data['next_observations'][i]
    next_obs = obs.reshape(48, 48, 3)


    combined_obs = np.concatenate((obs, next_obs), axis=1)
    combined_obs = combined_obs * 255.

    combined_obs = cv2.resize(combined_obs, (2*args.size, args.size))
#    combined_obs = cv2.cvtColor(combined_obs, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(args.out_path + "/observations_{}.png".format(i), combined_obs)



