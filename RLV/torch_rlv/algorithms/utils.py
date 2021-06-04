from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.algorithms.rlv.rlv import RLV


def init_algorithm(alg_name, agent, env, env_name, n_games, pre_steps):
    if alg_name == "sac":
        return SAC(env_name, env, agent, n_games=n_games, pre_steps=pre_steps)
    if alg_name == "rlv":
        return RLV(env_name, env, agent)
