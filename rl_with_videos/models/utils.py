from copy import deepcopy

from rl_with_videos.preprocessors.utils import get_preprocessor_from_params

def build_metric_learner_from_variant(variant, env, evaluation_data):
    sampler_params = variant['sampler_params']
    metric_learner_params = variant['metric_learner_params']
    metric_learner_params.update({
        'observation_shape': env.observation_space.shape,
        'max_distance': sampler_params['kwargs']['max_path_length'],
        'evaluation_data': evaluation_data
    })

    metric_learner = MetricLearner(**metric_learner_params)
    return metric_learner


def get_model_from_variant(variant, env, *args, **kwargs):
    pass

def get_inverse_model_from_variant(variant, env):
    print("observation space:", env.observation_space)
    print("shape:", env.observation_space.shape)
    print("action space:", env.action_space)
    print("shape:", env.action_space.shape)
    print("variant inverse model:", variant['inverse_model'])
    inverse_model_params = deepcopy(variant['inverse_model'])
    print("inverse model params:", inverse_model_params)

    if "Image" in variant['task'] and 'preprocessor_params' not in inverse_model_params:
        inverse_model_params.pop('hidden_layer_sizes')
        inverse_model_params.pop('inverse_domain_shift')
        from .convnet import convnet_model
        network = convnet_model(**inverse_model_params,
                                output_size=env.action_space.shape[0])

        return network,  None
    else:
        from .feedforward import feedforward_model
        preprocessor = None
        domain_shift_model = None
        if 'preprocessor_params' in inverse_model_params:
            preprocessor_params = inverse_model_params.pop('preprocessor_params', None)
            preprocessor = get_preprocessor_from_params(env, preprocessor_params)
            preprocessor = (preprocessor, preprocessor)

            if "inverse_domain_shift" in inverse_model_params:
                domain_shift_model = feedforward_model(hidden_layer_sizes=variant['inverse_model']['hidden_layer_sizes'],
                                                       input_shapes=(env.observation_space.shape,),
                                                       output_size=1,
                                                       stop_gradients=True,
                                                       preprocessors=preprocessor,
                                                       output_activation='sigmoid',
                                                       name="inverse_model_domain_shift_discriminator")
        network = feedforward_model(hidden_layer_sizes=variant['inverse_model']['hidden_layer_sizes'],
                                    input_shapes=(env.observation_space.shape, env.observation_space.shape),
                                    output_size=env.action_space.shape[0],
                                    preprocessors=preprocessor,
                                    name="inverse_model")
        return network, domain_shift_model


def flatten_input_structure(inputs):
    inputs_flat = nest.flatten(inputs)
    return inputs_flat

