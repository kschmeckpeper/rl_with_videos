import numpy.random
import torch.nn as nn
from RLV.torch_rlv.models.inverse_model import InverseModelNetwork
from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.buffer.provided_replay_pools.adapter_acrobot import get_acrobot_data
import numpy as np


def get_reward(done):
    if done == 1:
        return 10
    else:
        return -1


class RLV:
    def __init__(self, env_name, env, agent):
        super(RLV, self).__init__()
        self.env_name = env_name
        self.env = env
        self.agent = agent
        self.inverse_model = InverseModelNetwork(beta=0.0003, input_dims=6)
        self.iterations = 5

    def run(self):
        for _ in range(0, 20):  ## TODO
            # obs = self.env.reset()
            # action = self.agent.choose_action(obs)
            # observation_, reward, done, info = self.env.step(action)
            # self.agent.remember(obs, action, reward, observation_, done)
            #
            # state_int, action_int, reward_int, new_state_int, done_int = \
            #     self.agent.memory.sample_buffer(self.agent.batch_size)

            s, n, d, target = get_acrobot_data(_)
            self.agent.remember_action_free(s, n, d, target)

            state_obs, new_state_obs, done_obs, target = \
                self.agent.memory_action_free.sample_buffer(self.agent.batch_size)

            action_obs = self.inverse_model(state_obs)
            reward_obs = get_reward(done_obs)

            self.agent.remember(state_obs, action_obs, reward_obs, new_state_obs, done_obs)

            self.inverse_model.optimizer.zero_grad()
            loss = self.inverse_model.criterion(action_obs, target)
            loss.backward()
            self.inverse_model.optimizer.step()