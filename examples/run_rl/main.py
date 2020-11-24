import os
import copy
import glob
import pickle
import sys

import tensorflow as tf
import numpy as np
from ray import tune

from rl_with_videos.environments.utils import get_goal_example_environment_from_variant
from rl_with_videos.algorithms.utils import get_algorithm_from_variant
from rl_with_videos.policies.utils import get_policy_from_variant, get_policy
from rl_with_videos.replay_pools.utils import get_replay_pool_from_variant
from rl_with_videos.samplers.utils import get_sampler_from_variant
from rl_with_videos.value_functions.utils import get_Q_function_from_variant
from rl_with_videos.models.utils import get_inverse_model_from_variant
from rl_with_videos.misc.generate_goal_examples import get_goal_example_from_variant
from rl_with_videos.preprocessors.utils import get_preprocessor_from_params


from rl_with_videos.misc.utils import set_seed, initialize_tf_variables
from examples.instrument import run_example_local
from examples.development.main import ExperimentRunner

class ExperimentRunnerRL(ExperimentRunner):

    def _build(self):
        variant = copy.deepcopy(self._variant)


        training_environment = self.training_environment = (
            get_goal_example_environment_from_variant(variant))
        evaluation_environment = self.evaluation_environment = (
            get_goal_example_environment_from_variant(variant))

        if self._variant['shared_preprocessor']['use']:
            print("shared preprocessor:", variant['shared_preprocessor'])
            print("first preprocessor")
            shared_preprocessor = get_preprocessor_from_params(training_environment, variant['shared_preprocessor']['preprocessor_params'])
            print("shared_preprocessor:", shared_preprocessor)
            variant['inverse_model']['preprocessor_params']['shared_preprocessor_model'] = shared_preprocessor
            variant['Q_params']['kwargs']['preprocessor_params']['shared_preprocessor_model'] = shared_preprocessor
            variant['policy_params']['kwargs']['preprocessor_params']['shared_preprocessor_model'] = shared_preprocessor


        print("\n\n\n\nFinished env/shared preprocessor setup\n\n\n")
        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(variant)
        Qs = self.Qs = get_Q_function_from_variant(variant, training_environment)
        policy = self.policy = get_policy_from_variant(variant, training_environment, Qs)
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        algorithm_kwargs = {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'policy': policy,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler,
            'session': self._session,
        }

        if 'paired_data_pool' in variant and variant['paired_data_pool'] is not None:

            print("\n\n\n\n\nusing paired data pool\n\n\n\n\n\n")
            algorithm_kwargs['shared_preprocessor_model'] = shared_preprocessor
            algorithm_kwargs['paired_data_pool'] = get_replay_pool_from_variant(variant['paired_data_pool'], training_environment)
        
        print("algorithm type:", self._variant['algorithm_params']['type'])
        if self._variant['algorithm_params']['type'] in ['RLV']:
            action_free_replay_pool = get_replay_pool_from_variant(variant['action_free_replay_pool'], training_environment)
            algorithm_kwargs['action_free_pool'] = action_free_replay_pool
        if self._variant['algorithm_params']['type'] in ['RLV']:
            algorithm_kwargs['inverse_model'] = get_inverse_model_from_variant(variant, training_environment)
            print("inited replay pool and inverse model")

        self.algorithm = get_algorithm_from_variant(**algorithm_kwargs)

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def _restore(self, checkpoint_dir):
        assert isinstance(checkpoint_dir, str), checkpoint_dir

        checkpoint_dir = checkpoint_dir.rstrip('/')

        with self._session.as_default():
            pickle_path = self._pickle_path(checkpoint_dir)
            with open(pickle_path, 'rb') as f:
                picklable = pickle.load(f)

        training_environment = self.training_environment = picklable[
            'training_environment']
        evaluation_environment = self.evaluation_environment = picklable[
            'evaluation_environment']

        replay_pool = self.replay_pool = (
            get_replay_pool_from_variant(self._variant, training_environment))

        if self._variant['run_params'].get('checkpoint_replay_pool', False):
            self._restore_replay_pool(checkpoint_dir)

        sampler = self.sampler = picklable['sampler']
        Qs = self.Qs = picklable['Qs']
        # policy = self.policy = picklable['policy']
        policy = self.policy = (
            get_policy_from_variant(self._variant, training_environment, Qs))
        self.policy.set_weights(picklable['policy_weights'])
        initial_exploration_policy = self.initial_exploration_policy = (
            get_policy('UniformPolicy', training_environment))

        algorithm_kwargs = {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'policy': policy,
            'initial_exploration_policy': initial_exploration_policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler,
            'session': self._session,
        }


        print("algorithm type:", self._variant['algorithm_params']['type'])
        if self._variant['algorithm_params']['type'] in ['RLV']:
            print("does not currently restore inverse model or action free replay pool")
            raise NotImplementedError

        self.algorithm = get_algorithm_from_variant(**algorithm_kwargs)
        self.algorithm.__setstate__(picklable['algorithm'].__getstate__())

        tf_checkpoint = self._get_tf_checkpoint()
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            os.path.split(self._tf_checkpoint_prefix(checkpoint_dir))[0]))

        status.assert_consumed().run_restore_ops(self._session)
        initialize_tf_variables(self._session, only_uninitialized=True)

        # TODO(hartikainen): target Qs should either be checkpointed or pickled.
        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True

    @property
    def picklables(self):
        picklables = {
            'variant': self._variant,
            'training_environment': self.training_environment,
            'evaluation_environment': self.evaluation_environment,
            'sampler': self.sampler,
            'algorithm': self.algorithm,
            'Qs': self.Qs,
            'policy_weights': self.policy.get_weights(),
        }

        return picklables

def main(argv=None):
    """Run ExperimentRunner locally on ray.
    """
    # __package__ should be `development.main`
    run_example_local('examples.run_rl', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
