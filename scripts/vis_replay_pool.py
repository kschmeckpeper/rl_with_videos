import pickle
import gzip
import argparse

import cv2
import numpy as np
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('pool', type=str)
parser.add_argument('out_path', type=str)
args = parser.parse_args()

with gzip.open(args.pool, 'rb') as f:
    data = pickle.load(f)

print("Keys:", data.keys())

print("observations.shape:", data['observations'].shape)
#for i in tqdm.tqdm(range(1000)):
for i in tqdm.tqdm(range(0, data['observations'].shape[0], 1)):#, data['observations'].shape[0] // 1000)):
    if np.random.random() > 0.1:
        continue
    obs = data['observations'][i]
#    print("obs shape:", obs.shape)

    obs = obs.reshape(48, 48, 3)

#    print("obs dtype:", obs.dtype)
#    print("obs min,", np.min(obs), "max", np.max(obs))

    obs = obs * 255.

    cv2.imwrite(args.out_path + "/observation_{:06d}.png".format(i), obs)


    obs = data['next_observations'][i]
    obs = obs.reshape(48, 48, 3)

    obs = obs * 255.

#    cv2.imwrite(args.out_path + "/observation_{:06d}_next.png".format(i), obs)
