import torch as T
from RLV.torch_rlv.models.inverse_model import InverseModelNetwork
from RLV.torch_rlv.algorithms.sac.sac import SAC
import numpy as np


def set_reward(reward):
    if reward == -1:
        return -1
    else:
        return 10


class RLV:
    def __init__(self, env_name, env, agent):
        super(RLV, self).__init__()
        self.env_name = env_name
        self.env = env
        self.score_history = []
        self.agent = agent
        self.inverse_model = InverseModelNetwork(beta=0.0003, input_dims=6)
        self.iterations = 500  # TODO

    def run(self):
        p_steps = 100
        for x in range(0, self.iterations):
            observation = self.env.reset()
            for _ in range(0, self.agent.batch_size):
                act = self.agent.choose_action(observation)
                action = np.zeros(3)
                action[act] = 1
                observation_, reward, done, info = self.env.step(act)
                self.agent.remember_action_free(observation, action, reward, observation_, done)
                if done:
                    observation = self.env.reset()
                else:
                    observation = observation_

            state_obs, target, reward, next_state_obs, done_obs \
                = self.agent.memory_action_free.sample_buffer(self.agent.batch_size)

            input_inverse_model = T.from_numpy(state_obs)

            action_obs_t = self.inverse_model(input_inverse_model.float())

            reward_obs = np.zeros((self.agent.batch_size, 1))

            for __ in range(0, self.agent.batch_size):
                reward_obs[__] = set_reward(reward[__])

            done_obs = np.reshape(done_obs, (256, 1))

            action_obs = T.argmax(action_obs_t, dim=1).detach().numpy()

            for ___ in range(0, self.agent.batch_size):
                self.agent.remember(state_obs[___], action_obs[___],
                                    reward_obs[___], next_state_obs[___], done_obs[___])

            # s = SAC(env_name=self.env_name, env=self.env, agent=self.agent,
            #         n_games=1, pre_steps=p_steps, score_history=self.score_history)
            # s.run(cnt=x)
            #self.score_history = s.get_score_history()
            p_steps = 0


            target_t = T.from_numpy(target).float()
            self.inverse_model.optimizer.zero_grad()
            loss = self.inverse_model.criterion(action_obs_t, target_t)
            print(target_t)
            print(f"Iteration: {x} - Loss Inverse Model: {loss}")
            loss.backward()
            self.inverse_model.optimizer.step()
