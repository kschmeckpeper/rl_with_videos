import numpy.random
import torch.nn as nn
import torch as T
from RLV.torch_rlv.models.inverse_model import InverseModelNetwork
from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.buffer.provided_replay_pools.adapter_acrobot import get_acrobot_data
import numpy as np


def set_reward(done):
    if done:
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
        self.iterations = 1

    def run(self):
        for _ in range(0, self.iterations):  ## TODO
            st, next_st, term, target = get_acrobot_data(_ * 256, _ * 256 + 256)

            for __ in range(0, self.agent.batch_size):
                self.agent.remember_action_free(st[__], next_st[__], term[__], target[__])

            state_obs, next_state_obs, done_obs, target = \
                self.agent.memory_action_free.sample_buffer(self.agent.batch_size)

            predicted_action_obs = self.inverse_model(T.from_numpy(state_obs).float())
            set_reward_obs = np.zeros((self.agent.batch_size, 6))

            for __ in range(0, self.agent.batch_size):
                set_reward_obs[__] = set_reward(done_obs[__])

            state_obs = T.from_numpy(state_obs)
            next_state_obs = T.from_numpy(next_state_obs)
            set_reward_obs = T.from_numpy(set_reward_obs)
            done_obs = T.from_numpy(done_obs)
            target = T.from_numpy(target).float()

            #done_obs = T.reshape(done_obs, (256, 1))'

            # for __ in range(0, self.agent.batch_size):
            #     self.agent.remember(state_obs[__], predicted_action_obs[__],
            #                         set_reward_obs[__], next_state_obs[__], done_obs[__])

            # # Agent, Agent Funktion replace_replay_buffer

            # self.inverse_model.optimizer.zero_grad()
            # loss = self.inverse_model.criterion(predicted_action_obs, target)
            # loss.backward()
            # self.inverse_model.optimizer.step()
