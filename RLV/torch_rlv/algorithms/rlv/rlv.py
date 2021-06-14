import torch as T
from RLV.torch_rlv.models.inverse_model import InverseModelNetwork
from RLV.torch_rlv.algorithms.sac.sac import SAC
from RLV.torch_rlv.visualizer.plot import plot_learning_curve, plot_env_step, animate_env_obs
import numpy as np
from datetime import datetime
import wandb


def set_reward(reward):
    if reward == -1:
        return -1
    else:
        return 10


class RLV:
    def __init__(self, env_name, env, agent, iterations=500, warmup_steps=500, base_algorithm=None, lr=0.003,
                 experiment_name='RLV', pre_steps=1000, pre_training_steps=25000):
        super(RLV, self).__init__()
        self.experiment_name=experiment_name
        self.env_name = env_name
        self.env = env
        self.steps_count = 0
        self.lr = lr
        self.pre_steps = pre_steps
        self.warmup_steps = warmup_steps
        self.pre_training_steps = pre_training_steps
        self.score_history = []
        self.agent = agent
        self.inverse_model = InverseModelNetwork(beta=0.0003, input_dims=13)
        self.iterations = iterations  # TODO
        self.filename = env_name + '.png'
        self.figure_file = 'output/plots/RLV_' + self.filename
        self.date_time = datetime.now().strftime("%m_%d_%Y_%H:%M")
        self.algorithm = base_algorithm

    def get_action_free_buffer(self):
        steps = 0
        pre_training = SAC(env_name=self.env_name, env=self.env, agent=self.agent,
                           n_games=1, pre_steps=100, score_history=self.score_history,
                           additional_data=None, steps_count=0,
                           lr=self.lr, rlv_config=None, experiment_name=self.experiment_name)
        while steps < self.pre_training_steps:
            pre_training.run()
            steps = pre_training.steps_count
        return pre_training.agent.memory

    def warmup_inverse_model(self, warmup_steps):
        for iter in range(0, warmup_steps):
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

            target_t = T.from_numpy(target).float()
            self.inverse_model.optimizer.zero_grad()
            loss = self.inverse_model.criterion(action_obs_t, target_t)

            if iter % 50 == 0:
                print(f"Warmup Step: {iter} - Loss Inverse Model: {loss}")

            # Update Inverse Model
            loss.backward()
            self.inverse_model.optimizer.step()

    def run(self, plot=False):
        p_steps = self.pre_steps
        self.agent.memory_action_free = self.get_action_free_buffer()

        self.warmup_inverse_model(warmup_steps=self.warmup_steps)

        for iter in range(0, self.iterations):
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

            # Inverse Model
            target_t = T.from_numpy(target).float()
            self.inverse_model.optimizer.zero_grad()
            loss = self.inverse_model.criterion(action_obs_t, target_t)
            print(f"Iteration: {iter} - Loss Inverse Model: {loss}")

            rlv_args = {
                'experiment_name': 'rlv_exp_' + str(self.lr),
                'loss_inverse_model': loss,
                'warmup_steps': self.warmup_steps
            }

            # perform sac based on initial data obtained by environment step plus additional
            # observational data
            if self.algorithm is None:
                self.algorithm = SAC(env_name=self.env_name, env=self.env, agent=self.agent,
                                     n_games=1, pre_steps=p_steps, score_history=self.score_history,
                                     additional_data=observational_batch, steps_count=self.steps_count,
                                     lr=self.lr, rlv_config=rlv_args, experiment_name=self.experiment_name)
            # execute pre steps only in first iteration
            if iter > 0:
                self.algorithm.run(cnt=iter, execute_pre_steps=False)
            else:
                self.algorithm.run(cnt=iter)

            # update steps count of RLV based on steps executed in SAC
            self.steps_count = self.algorithm.get_step_count()
            self.score_history = self.algorithm.get_score_history()

            p_steps = 0

            # Update Inverse Model
            loss.backward()
            self.inverse_model.optimizer.step()

        # Plot in pdf file with visualizer
        if plot:
            env_states = self.algorithm.get_env_states()
            plot_env_step(env_states, 'output/plots/SAC_' + self.env_name
                          + '_' + self.date_time)
            observations = self.algorithm.get_env_obs()
            animate_env_obs(observations, 'output/videos/RLV_' + self.env_name + '_' + self.date_time)
            x = [i + 1 for i in range(len(self.score_history))]
            plot_learning_curve(x, self.score_history, self.figure_file)
