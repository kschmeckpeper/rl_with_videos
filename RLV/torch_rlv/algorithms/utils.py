import numpy as np
import matplotlib.pyplot as plt
from RLV.torch_rlv.algorithms.sac.sac import SAC


def init_algorithm(alg_name, agent, env, env_name, n_actions, pre_steps):
    if alg_name == "sac":
        return SAC(env_name, env, agent, n_games=n_actions, pre_steps=pre_steps)


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
