from ..environments.utils import get_environment
from ..algorithms.utils import init_algorithm
from ..algorithms.sac.sac_agent import Agent

# class
# argument discrete, continous
# getAgen


def run_experiment(env_name, algo_name, n_actions, pre_steps, layer1_size, layer2_size, lr):
    env = get_environment(env_name)
    agent = Agent(alpha=lr, beta=lr, input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0], layer1_size=layer1_size, layer2_size=layer2_size)
    algorithm = init_algorithm(algo_name, agent, env, env_name, n_actions=n_actions, pre_steps=pre_steps)

    algorithm.run()


