from RLV.torch_rlv.executor.experiment import Experiment


def run_torch_rlv(config):
    acrobot_test_experiment = Experiment(config)
    acrobot_test_experiment.run_experiment()


run_torch_rlv({
    'action_space_type': 'discrete',
    'env_name': 'acrobot',
    'algo_name': 'algo_name',
    'n_actions': 250,
    'pre_steps': 1000,
    'layer1_size': 256,
    'layer2_size': 256,
    'lr': 0.003
})
