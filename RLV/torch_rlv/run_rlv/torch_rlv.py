from RLV.torch_rlv.executor.experiment import Experiment


def main():
    acrobot_test_config = {
        'action_space_type': 'discrete',
        'env_name': 'acrobot',
        'algo_name': 'algo_name',
        'n_actions': 2500,
        'pre_steps': 1000,
        'layer1_size': 256,
        'layer2_size': 256,
        'lr': 0.003
    }
    acrobot_test_experiment = Experiment(acrobot_test_config)
    acrobot_test_experiment.run_experiment()


if __name__ == "__main__":
    main()
