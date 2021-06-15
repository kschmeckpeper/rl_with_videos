import numpy as np
from RLV.torch_rlv.visualizer.plot import plot_env_step, animate_env_obs
from datetime import datetime
import wandb


class SAC:
    def __init__(self, env_name, env, agent, n_games=2500, load_checkpoint=False, steps=100,
                 pre_steps=100, steps_count=0, plot_steps=250, lr=0.003, score_history=None,
                 additional_data=None, project='NEW_225k_sac_rlv', rlv_config=None, experiment_name='SAC'):
        super(SAC, self).__init__()
        if score_history is None:
            score_history = []
        self.env_name = env_name
        self.experiment_name = experiment_name
        self.env = env
        self.agent = agent
        self.lr = lr
        self.pre_steps = pre_steps
        self.steps = steps
        self.project = project
        self.steps_count = steps_count
        self.plot_steps = plot_steps
        self.n_games = n_games
        self.load_checkpoint = load_checkpoint
        self.filename = env_name + '.png'
        self.additional_data = additional_data
        self.figure_file = 'output/plots/SAC' + self.filename
        self.date_time = datetime.now().strftime("%m_%d_%Y_%H:%M")

        # best score, score_history to calculate average score
        self.best_score = env.reward_range[0]
        self.score_history = score_history
        self.env_states = []
        self.env_obs = []

        # if SAC is used in RLV, log additional parameters which are in rlv_config in wandb
        self.rlv_config = rlv_config

    def run(self, cnt=-1, plot=False, execute_pre_steps=True):
        run_name = self.experiment_name + str(self.lr)
        wandb.init(project=self.project, name=run_name)

        # initial logging parameters when SAC is used
        logging_parameters = {'env_name': self.env_name,
                              'pre_steps': self.pre_steps,
                              'steps_count': self.steps_count,
                              'plot_steps': self.plot_steps,
                              'n_games': self.n_games,
                              'score_history': self.score_history,
                              'learning_rate': self.lr}

        # if SAC is used in RLV, additional parameters of RLV are logged in wandb
        if self.rlv_config is not None:
            logging_parameters.update(self.rlv_config)
        wandb.log(logging_parameters)

        if self.load_checkpoint:
            self.agent.load_models()
            self.env.render(mode='human')

        obs = self.env.reset()

        if execute_pre_steps:
            for _ in range(self.pre_steps):
                action = self.env.action_space.sample()
                obs_, reward, done, info = self.env.step(action)
                action = np.eye(self.env.action_space.n)[action]
                self.agent.remember(obs, action, reward, obs_, done)
                obs = obs_

        for i in range(self.n_games):
            observation = self.env.reset()
            done = False
            score = 0
            step = 0
            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.agent.remember(observation, np.eye(self.env.action_space.n)[action], reward, observation_, done)
                if not self.load_checkpoint:
                    self.agent.learn(mixed_pool=self.additional_data)

                if plot:
                    obs_img = self.env.render(mode='rgb_array')
                    self.env_obs.append(obs_img)
                    if step % self.plot_steps == 0 or done:
                        self.env_states.append(obs_img)

                observation = observation_
                step += 1

            # increase steps count, update score_history and calculate avg_score
            self.steps_count += step
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])

            if avg_score > self.best_score:
                self.best_score = avg_score
                if not self.load_checkpoint:
                    self.agent.save_models()
            if cnt != -1:
                i = cnt
            wandb.log({'steps': self.steps_count,
                       'episode': i,
                       'best score': self.best_score,
                       'score': score,
                       'avg_score': avg_score})
            print('steps', self.steps_count, ' episode ', i, ' best score %.1f' % self.best_score,
                  ' score %.1f' % score, ' avg_score %.1f' % avg_score)

        if plot:
            plot_env_step(self.env_states, 'output/plots/SAC_' + self.env_name
                          + '_' + self.date_time)
            animate_env_obs(self.env_obs, 'output/videos/SAC_' + self.env_name
                            + '_' + self.date_time)

    def get_score_history(self):
        return self.score_history

    def get_step_count(self):
        return self.steps_count

    def get_env_states(self):
        return self.env_states

    def get_env_obs(self):
        return self.env_obs
