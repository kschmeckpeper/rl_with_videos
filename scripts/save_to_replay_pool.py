import gzip
import pickle
import argparse
from os.path import isfile, join
from os import listdir
import numpy as np
import cv2

def save_replay_pool_from_images(args):
    files = [f for f in listdir(args.image_path) if isfile(join(args.image_path, f))]

    trajectories = {}
    print("Files:", len(files))
    for f in files:
#        print("f:", f)
        if "fake_B" in f:
            traj_name = f[:-15]

            index = int(f[-15:-11])
        elif "frame" in f and "video" in f:
            sections = f.split('_')
#            print("sections", sections)
            if len(sections) > 4:
                traj_name = '_'.join(sections[:3])
                index = int(sections[4][:-4])
            else:
                traj_name = sections[1]
                index = int(sections[3][:-4])
#            print("traj_name", traj_name, "index", index)
        else:
            sections = f.split('_')
            traj_name = sections[1]
            index = int(sections[2][:-4])
#            print("traj_name, index", traj_name, index)
#            continue

        if traj_name not in trajectories:
            trajectories[traj_name] = {}
        trajectories[traj_name][index] = f



    data = {'observations': [],
            'next_observations': [],
            'rewards': [],
            'terminals': [],
            'actions': []
            }

    for traj in trajectories:
        print("traj:", traj)
        indices = sorted(trajectories[traj].keys())
        print("indices", indices)
        if args.reverse:
            indices = indices[::-1]
        print("reversed_indices", indices)
        next_im = cv2.imread(join(args.image_path, trajectories[traj][indices[0]]))
        next_im = cv2.resize(next_im, args.image_size)
        next_im = next_im.reshape(-1)/255.0
        for i in range(len(indices) - 1):
            if i % args.save_fraction != 0:
                continue
            data['observations'].append(next_im)
            next_im = cv2.imread(join(args.image_path, trajectories[traj][indices[i]]))
            next_im = cv2.resize(next_im, args.image_size)
            next_im = next_im.reshape(-1)/255.0
            data['next_observations'].append(next_im)
            data['rewards'].append(0.0)
            data['terminals'].append(1 if i == len(indices)-2 else 0)
            data['actions'].append([0.0 for _ in range(args.action_space)])
        data['terminals'][-1] = 1
    
    for k in data:
        data[k] = np.array(data[k])

    data['rewards'] = data['rewards'].reshape(-1, 1)
    data['terminals'] = data['terminals'].reshape(-1, 1)

    for k in data:
        print("k,", k, data[k].shape)

    with gzip.open(args.out_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument("out_path", type=str)
    parser.add_argument("--action_space", type=int, default=2)
    parser.add_argument("--image_size", default=(48, 48))
    parser.add_argument('--save_fraction', default=1, type=int)
    parser.add_argument('--reverse', action='store_true', dest='reverse')
    args = parser.parse_args()

    save_replay_pool_from_images(args)
