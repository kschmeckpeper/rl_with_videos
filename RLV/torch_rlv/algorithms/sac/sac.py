import numpy as np
from .. import utils


class SAC:
    def __init__(self, env_name, env, agent, n_games=2500, load_checkpoint=False, steps=100,
                 pre_steps=100, steps_count=0,
                 score_history=[], additional_data=None):
        super(SAC, self).__init__()
        self.env = env
        self.agent = agent
        self.pre_steps = pre_steps
        self.steps = steps
        self.steps_count = steps_count
        self.n_games = n_games
        self.load_checkpoint = load_checkpoint
        self.filename = env_name + '.png'
        self.additional_data = additional_data

        self.figure_file = 'output/plots/' + self.filename

        self.best_score = env.reward_range[0]
        self.score_history = score_history

    def run(self, cnt=-1, plot=False):

        if self.load_checkpoint:
            self.agent.load_models()
            self.env.render(mode='human')

        obs = self.env.reset()
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
            s = 0
            while not done:
                action = self.agent.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.agent.remember(observation, np.eye(self.env.action_space.n)[action], reward, observation_, done)
                if not self.load_checkpoint:
                    self.agent.learn(mixed_pool=self.additional_data)
                observation = observation_
                s += 1
            self.steps_count += s
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])

            if avg_score > self.best_score:
                self.best_score = avg_score
                if not self.load_checkpoint:
                    self.agent.save_models()
            if cnt != -1:
                i = cnt
            print('steps', self.steps_count, ' episode ', i, ' best score %.1f' % self.best_score,
                  ' score %.1f' % score, ' avg_score %.1f' % avg_score)

        if plot:
            if not self.load_checkpoint:
                x = [i + 1 for i in range(self.n_games)]
                utils.plot_learning_curve(x, self.score_history, self.figure_file)

    def get_score_history(self):
        return self.score_history
