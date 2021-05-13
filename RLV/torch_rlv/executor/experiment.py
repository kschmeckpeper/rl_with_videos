from ..environments.utils import get_environment
from ..algorithms.utils import init_algorithm
from ..algorithms.sac.sac_agent import Agent

from typing import Dict, Any

# dictionary as input
# getAgent


class Experiment:
    def __init__(self, config):
        self.config = config,
        self.action_space_type = config['action_space_type'],
        self.env_name = config['env_name'],
        self.algo_name = config['algo_name'],
        self.n_actions = config['n_actions'],
        self.pre_steps = config['pre_steps'],
        self.layer1_size = config['layer1_size'],
        self.layer2_size = config['layer2_size'],
        self.lr = config['lr']

    def run_experiment(self):
        env = get_environment(self.env_name)
        agent = Agent(alpha=self.lr, beta=self.lr, input_dims=env.observation_space.shape, env=env,
                      n_actions=env.action_space.shape[0], layer1_size=self.layer1_size, layer2_size=self.layer2_size)
        algorithm = init_algorithm(self.algo_name, agent, env, self.env_name, n_actions=self.n_actions,
                                   pre_steps=self.pre_steps)

        algorithm.run()
