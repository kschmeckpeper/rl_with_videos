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
        self.iterations = 20  # TODO

    def run(self):
        for _ in range(0, self.iterations):
            state, next_state, terminal, target = get_acrobot_data(_ * 256, _ * 256 + 256)

            for __ in range(0, self.agent.batch_size):
                self.agent.remember_action_free(state[__], next_state[__], terminal[__], target[__])

            state_obs, next_state_obs, target, done_obs = \
                self.agent.memory_action_free.sample_buffer(self.agent.batch_size)

            input_inverse_model = T.from_numpy(state_obs)

            predicted_action_obs_t = self.inverse_model(input_inverse_model.float())
            reward_obs = np.zeros((self.agent.batch_size, 1))

            for __ in range(0, self.agent.batch_size):
                reward_obs[__] = set_reward(done_obs[__])

            done_obs = np.reshape(done_obs, (256, 1))

            predicted_action_obs = predicted_action_obs_t.detach().numpy()

            for __ in range(0, self.agent.batch_size):
                self.agent.remember(state_obs[__], predicted_action_obs[__],
                                    reward_obs[__], next_state_obs[__], done_obs[__])

            s = SAC(env_name=self.env_name, env=self.env, agent=self.agent,
                    n_games=1, pre_steps=0)
            s.run(cnt=_)

            target_t = T.from_numpy(target).float()

            self.inverse_model.optimizer.zero_grad()
            loss = self.inverse_model.criterion(predicted_action_obs_t, target_t)
            print(f"Iteration: {_} - Loss Inverse Model: {loss}")

            loss.backward()
            self.inverse_model.optimizer.step()
