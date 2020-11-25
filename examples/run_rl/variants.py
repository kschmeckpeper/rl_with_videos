from ray import tune
import numpy as np

from rl_with_videos.misc.utils import get_git_rev, deep_update
from rl_with_videos.misc.generate_goal_examples import DOOR_TASKS, PUSH_TASKS, PICK_TASKS

M = 256
#M = 400
M_layers = 2
#M_layers = 4
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M,) * M_layers,
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 100
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'DAPG': 1000,
    'Metaworld': 200,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 5,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}

ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'n_epochs': 200,
        }
    },
    'RLV': {
        'type': 'RLV',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'n_epochs': 200,
        }
    },
}

DEFAULT_NUM_EPOCHS = 200
NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    # algorithm_params = deep_update(
    #     ALGORITHM_PARAMS_BASE,
    #     ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    # )
    # algorithm_params = deep_update(
    #     algorithm_params,
    #     ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    # )
    algorithm_params = ALGORITHM_PARAMS_BASE
    algorithm_params = deep_update(
            algorithm_params,
            ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
        )

    variant_spec = {
        'domain': domain,
        'task': task,
        'universe': universe,
        'git_sha': get_git_rev(),

        #'env_params': ENV_PARAMS.get(domain, {}).get(task, {}),
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M,) * M_layers,
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': 1e6,
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency':  DEFAULT_NUM_EPOCHS // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec



def get_variant_spec(args):
    universe, domain = args.universe, args.domain
    task, algorithm, n_epochs = args.task, args.algorithm, args.n_epochs

    variant_spec = get_variant_spec_base(
        universe, domain, task, args.policy, args.algorithm)

    if args.algorithm in ('RLV',):
        variant_spec['action_free_replay_pool'] = {
            'replay_pool_params': {
                'type': 'ActionFreeReplayPool',
                'kwargs': {
                    'max_size': 1e6,
                    'data_path': args.replay_pool_load_path,
                    'remove_rewards': args.remove_rewards,
                    'max_demo_length': args.max_demo_length,
                    'use_ground_truth_actions': args.use_ground_truth_actions,
                }
            }
        }
        if args.paired_data_path is not None:
            variant_spec['paired_data_pool'] = {
                    'replay_pool_params': {
                        'type': 'ActionFreeReplayPool',
                        'kwargs': {
                            'max_size': 1e6,
                            'data_path': args.paired_data_path,
                            'remove_rewards': True,
                            'max_demo_length': args.max_demo_length,
                            'use_ground_truth_actions': False,
                        }
                    }
                }
        else:
            variant_spec['paired_data_pool'] = None
        variant_spec['algorithm_params']['kwargs']['remove_rewards'] = args.remove_rewards
        variant_spec['algorithm_params']['kwargs']['replace_rewards_scale'] = args.replace_rewards_scale
        variant_spec['algorithm_params']['kwargs']['replace_rewards_bottom'] = args.replace_rewards_bottom
        variant_spec['algorithm_params']['kwargs']['use_ground_truth_actions'] = args.use_ground_truth_actions
        variant_spec['algorithm_params']['kwargs']['use_zero_actions'] = args.use_zero_actions
        variant_spec['algorithm_params']['kwargs']['paired_loss_scale'] = args.paired_loss_scale
        if args.algorithm in ('RLV'):
            variant_spec['inverse_model'] = {
                'hidden_layer_sizes': [64, 64, 64],
                'conv_filters': [64] * 5,
                'conv_kernel_sizes': [3] * 5,
                'conv_strides': [2] * 5,
                'inverse_domain_shift': args.inverse_domain_shift,
            }
            variant_spec['algorithm_params']['kwargs']['preprocessor_for_inverse'] = args.preprocessor_for_inverse
            variant_spec['algorithm_params']['kwargs']['inverse_domain_shift'] = args.inverse_domain_shift
            variant_spec['algorithm_params']['kwargs']['inv_model_ds_generator_weight'] = args.inv_model_ds_generator_weight
            variant_spec['algorithm_params']['kwargs']['inv_model_ds_discriminator_weight'] = args.inv_model_ds_discriminator_weight



    variant_spec['algorithm_params']['kwargs']['n_epochs'] = \
            n_epochs

    variant_spec['algorithm_params']['kwargs']['should_augment'] = False
    variant_spec['algorithm_params']['kwargs']['trans_dist'] = args.trans_dist
    variant_spec['algorithm_params']['kwargs']['n_train_repeat'] = args.n_train_repeat


    if 'Image48' in task:
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                #'image_shape': variant_spec['env_params']['image_shape'],
                'image_shape': (48, 48, 3),
                'output_size': M,
                'conv_filters': (8, 8),
                'conv_kernel_sizes': ((5, 5), (5, 5)),
                'pool_type': 'MaxPool2D',
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        if args.preprocessor_for_inverse:
            preprocessor_params['kwargs']['conv_filters'] = [16, 16, 32] #variant_spec['inverse_model']['conv_filters']
            preprocessor_params['kwargs']['pool_strides'] = [2]*3#variant_spec['inverse_model']['conv_strides']
            preprocessor_params['kwargs']['conv_kernel_sizes'] = [5]*3#variant_spec['inverse_model']['conv_kernel_sizes']
            preprocessor_params['kwargs']['pool_sizes'] = [(2, 2)] * 3
            variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (2*M,) + (M,)*M_layers
            variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (2*M,) + (M,)*M_layers
#            variant_spec['inverse_model']['hidden_layer_sizes'] = (2*M, M, M)
            if 'inverse_model' in variant_spec:
                variant_spec['inverse_model']['preprocessor_params'] = preprocessor_params.copy()

        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['replay_pool_params']['kwargs']['max_size'] = min(1000000, int(n_epochs*1000))


        variant_spec['shared_preprocessor'] = {'use': args.shared_preprocessor,
                                               'preprocessor_params': preprocessor_params.copy()
                                               }
        
        variant_spec['algorithm_params']['kwargs']['should_augment'] = True


    elif 'Image' in task:
        raise NotImplementedError('Add convnet preprocessor for this image input')
    else:
        variant_spec['shared_preprocessor'] = {'use': False}

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
