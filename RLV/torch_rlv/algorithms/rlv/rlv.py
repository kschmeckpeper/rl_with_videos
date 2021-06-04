import torch as T
from RLV.torch_rlv.models.inverse_model import InverseModelNetwork
from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.visualizer.plot import plot_learning_curve, plot_env_step
import numpy as np


def set_reward(reward):
    if reward == -1:
        return -1
    else:
        return 10


class RLV:
    def __init__(self, env_name, env, agent, iterations=500, warm_up_steps=500):
        super(RLV, self).__init__()
        self.env_name = env_name
        self.env = env
        self.steps_count = 0
        self.warmup_steps = warm_up_steps
        self.score_history = []
        self.agent = agent
        self.inverse_model = InverseModelNetwork(beta=0.0003, input_dims=13)
        self.iterations = iterations  # TODO
        self.filename = env_name + '.png'
        self.figure_file = 'output/plots/SAC' + self.filename

    def fill_action_free_buffer(self):
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

    def warmup_inverse_model(self, warmup_steps):
        for x in range(0, warmup_steps):
            state_obs, target, reward, next_state_obs, done_obs \
                = self.agent.memory_action_free.sample_buffer(self.agent.batch_size)
            done_obs = np.reshape(done_obs, (256, 1))

            # get actions and rewards for observational data
            input_inverse_model = T.cat((T.from_numpy(state_obs), T.from_numpy(next_state_obs),
                                         T.from_numpy(done_obs)), dim=1).float()

            action_obs_t = self.inverse_model(input_inverse_model)
            reward_obs = np.zeros((self.agent.batch_size, 1))
            for __ in range(0, self.agent.batch_size):
                reward_obs[__] = set_reward(reward[__])

            # Update Inverse Model
            target_t = T.from_numpy(target).float()
            self.inverse_model.optimizer.zero_grad()
            loss = self.inverse_model.criterion(action_obs_t, target_t)
            print(f"Warmup Step: {x} - Loss Inverse Model: {loss}")
            loss.backward()
            self.inverse_model.optimizer.step()

    def run(self, plot=False):
        p_steps = 100
        for _ in range(0, self.iterations):
            self.fill_action_free_buffer()

        self.warmup_inverse_model(warmup_steps=self.warmup_steps)

        for x in range(0, self.iterations):
            state_obs, target, reward, next_state_obs, done_obs \
                = self.agent.memory_action_free.sample_buffer(self.agent.batch_size)
            done_obs = np.reshape(done_obs, (256, 1))

            # get actions and rewards for observational data
            input_inverse_model = T.cat((T.from_numpy(state_obs), T.from_numpy(next_state_obs),
                                         T.from_numpy(done_obs)), dim=1).float()

            action_obs_t = self.inverse_model(input_inverse_model)

            reward_obs = np.zeros((self.agent.batch_size, 1))
            for __ in range(0, self.agent.batch_size):
                reward_obs[__] = set_reward(reward[__])

            # define observational data
            action_obs = action_obs_t.detach().numpy()

            observational_batch = {
                'state': state_obs,
                'action': action_obs,
                'reward': reward_obs,
                'next_state': next_state_obs,
                'done_obs': done_obs
            }

            # perform sac based on initial data obtained by environment step plus additional
            # observational data
            algorithm = SAC(env_name=self.env_name, env=self.env, agent=self.agent,
                            n_games=1, pre_steps=p_steps, score_history=self.score_history,
                            additional_data=observational_batch, steps_count=self.steps_count)
            algorithm.run(cnt=x)
            self.steps_count = algorithm.get_step_count()
            self.score_history = algorithm.get_score_history()
            env_step = algorithm.get_env_state()
            plot_env_step(env_step, self.steps_count, 'output/videos/RLV_' + self.env_name)
            p_steps = 0

            # Update Inverse Model
            target_t = T.from_numpy(target).float()
            self.inverse_model.optimizer.zero_grad()
            loss = self.inverse_model.criterion(action_obs_t, target_t)
            print(f"Iteration: {x} - Loss Inverse Model: {loss}")
            loss.backward()
            self.inverse_model.optimizer.step()

        if plot:
            x = [i + 1 for i in range(len(self.score_history))]
            plot_learning_curve(x, self.score_history, self.figure_file)
