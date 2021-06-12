from ..environments.utils import get_environment
from ..algorithms.utils import init_algorithm
from ..algorithms.sac.sac_agent import get_agent


# dictionary as input
# getAgent

class Experiment:
    def __init__(self, config):
        self.config = config
        self.action_space_type = config['action_space_type']
        self.env_name = config['env_name']
        self.algo_name = config['algo_name']
        self.n_games = config['n_games']
        self.pre_steps = config['pre_steps']
        self.warmup_steps = config['warmup_steps']
        self.layers = config['layers']
        self.lr = config['lr']

    def run_experiment(self):
        env = get_environment(self.env_name)
        agent = get_agent(env, self.action_space_type,  self)
        algorithm = init_algorithm(self.algo_name, agent, env, self.env_name, n_games=self.n_games,
                                   pre_steps=self.pre_steps, warmup_steps=self.warmup_steps)

        algorithm.run(plot=True)
        env.close()
