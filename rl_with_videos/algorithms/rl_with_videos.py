import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np
 
from collections import OrderedDict


from rl_with_videos.algorithms.sac import SAC
from rl_with_videos.models.utils import flatten_input_structure


class RLV(SAC):
    def __init__(self,
                 remove_rewards=False,
                 replace_rewards_scale=1.0,
                 replace_rewards_bottom=0.0,
                 use_ground_truth_actions=False,
                 use_zero_actions=False,
                 preprocessor_for_inverse=False,
                 inverse_domain_shift=False,
                 inv_model_ds_generator_weight=0.01,
                 inv_model_ds_discriminator_weight=0.01,
                 paired_loss_scale=1.0,
                 paired_data_pool=None,
                 shared_preprocessor_model=None,
                 **kwargs):
        print("\n\n\n\n\nkwargs in rlv:", kwargs)
        print("\n\n\n\n\n\n")
        print("paired_data pool", paired_data_pool)
        print("shared preprocessor model", shared_preprocessor_model)
        self._paired_data_pool = paired_data_pool
        self._paired_loss_scale = paired_loss_scale
        self._shared_preprocessor_model = shared_preprocessor_model

        self._action_free_pool = kwargs.pop('action_free_pool')
        self._inverse_model, self._inverse_domain_model = kwargs.pop('inverse_model')
        self._inverse_model_lr = 3e-4
        self._inverse_model_discrim_lr = 3e-4

        self._paired_loss_lr = 3e-4

        self._inverse_domain_shift = inverse_domain_shift
        self._inverse_model_domain_shift_generator_weight = inv_model_ds_generator_weight
        self._inverse_model_domain_shift_discriminator_weight = inv_model_ds_discriminator_weight

        self._remove_rewards = remove_rewards
        self._replace_rewards_scale = replace_rewards_scale
        self._replace_rewards_bottom = replace_rewards_bottom

        self._use_ground_truth_actions = use_ground_truth_actions
        self._use_zero_actions = use_zero_actions

        self._preprocessor_for_inverse = preprocessor_for_inverse


        super(RLV, self).__init__(**kwargs)

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()

        self._init_augmentation()

        if self._remove_rewards:
            self._init_reward_generation()

        self._init_inverse_model()

        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _init_placeholders(self):
        action_conditioned_placeholders = {
            'observations_no_aug': tf.placeholder(tf.float32,
                                           shape=(None, *self._observation_shape),
                                           name="observation_no_aug")
            ,
            'next_observations_no_aug': tf.placeholder(tf.float32,
                                                shape=(None, *self._observation_shape),
                                                name="next_observation_no_aug"),
            'actions': tf.placeholder(
                dtype=tf.float32,
                shape=(None, *self._action_shape),
                name='actions',
            ),
            'rewards': tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='rewards',
            ),
            'terminals': tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='terminals',
            ),
            'iteration': tf.placeholder(
                tf.int64, shape=(), name='iteration',
            ),
        }
        action_free_placeholders = {
            'observations_no_aug': tf.placeholder(tf.float32,
                                           shape=(None, *self._observation_shape),
                                           name="observation_no_aug")
            ,
            'next_observations_no_aug': tf.placeholder(tf.float32,
                                                shape=(None, *self._observation_shape),
                                                name="next_observation_no_aug"),
            'rewards': tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='rewards',
            ),
            'terminals': tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='terminals',
            ),
            'iteration': tf.placeholder(
                tf.int64, shape=(), name='iteration',
            ),
        }
        if self._remove_rewards:
            action_free_placeholders.pop('rewards')
        if self._use_ground_truth_actions:
            action_free_placeholders['actions'] = tf.placeholder(
                dtype=tf.float32,
                shape=(None, *self._action_shape),
                name='actions',
            )


        self._placeholders = {
                              'action_free': action_free_placeholders,
                              'action_conditioned': action_conditioned_placeholders
                            }


        if self._paired_data_pool is not None:
            self._placeholders['paired_data'] = {
                'obs_of_observation_no_aug': tf.placeholder(tf.float32,
                                           shape=(None, *self._observation_shape),
                                           name="obs_of_observation_no_aug"),
                'obs_of_interaction_no_aug': tf.placeholder(tf.float32,
                                           shape=(None, *self._observation_shape),
                                           name="obs_of_interaction_no_aug")
                }

        if self._domain_shift or self._inverse_domain_shift:
            self._domains_ph = tf.placeholder(tf.float32, shape=(None, 1), name='domains')



    def _training_batch(self, batch_size=None):
        batch = self.sampler.random_batch(batch_size)
        action_free_batch_size = 256
        action_free_batch = self._action_free_pool.random_batch(action_free_batch_size)
        combined_batch = {
                'action_conditioned': batch,
                'action_free': action_free_batch
            }
        if self._paired_data_pool is not None:
            paired_data_batch_size = 256
            combined_batch['paired_data'] = self._paired_data_pool.random_batch(paired_data_batch_size)
        return combined_batch

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""
        
        

        feed_dict = {}
        for action in ['action_conditioned', 'action_free']:
            for k in batch[action].keys():
                if k in ['observations', 'next_observations']:
                    feed_dict[self._placeholders[action][k+"_no_aug"]] = batch[action][k]
                else:
                    feed_dict[self._placeholders[action][k]] = batch[action][k]
        if iteration is not None:
            feed_dict[self._placeholders['action_conditioned']['iteration']] = iteration

        if self._domain_shift or self._inverse_domain_shift:
            feed_dict[self._domains_ph] = np.concatenate([np.zeros(batch['action_conditioned']['terminals'].shape),
                                                  np.ones(batch['action_free']['terminals'].shape)])


        if self._paired_data_pool is not None:
            feed_dict[self._placeholders['paired_data']['obs_of_observation_no_aug']] = batch['paired_data']['observations']
            feed_dict[self._placeholders['paired_data']['obs_of_interaction_no_aug']] = batch['paired_data']['next_observations']
        return feed_dict

    def _init_augmentation(self):
        top_level_keys = ['action_conditioned', 'action_free']
        if 'paired_data' in self._placeholders.keys():
            top_level_keys.append('paired_data')
            print("\n\n\n\naugmenting paired data\n\n\n\n")
        for action in top_level_keys:
            keys = list(self._placeholders[action].keys())
            for k in keys:
                if k[-7:] == '_no_aug':
                    print("augmenting", action, k)
                    self._placeholders[action][k[:-7]] = self._augment_image(self._placeholders[action][k])

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        print("Diagnostics ops:", self._diagnostics_ops)
        diagnostics = self._session.run(self._diagnostics_ops, feed_dict)

        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(batch['action_conditioned']['observations']).items()
        ]))

        if self._plotter:
            self._plotter.draw()

        return diagnostics



    def _init_reward_generation(self):
        print("Removed rewards.  Running reward generation")
        self._placeholders['action_free']['rewards'] = tf.math.multiply(self._replace_rewards_scale,
                                                                        tf.cast(self._placeholders['action_free']['terminals'], dtype=tf.float32))
        self._placeholders['action_free']['rewards'] += tf.math.multiply(self._replace_rewards_bottom,
                                                                         1.0 - tf.cast(self._placeholders['action_free']['terminals'], dtype=tf.float32))

    def _init_inverse_model(self):
        """ Creates minimization ops for inverse model.

        Creates a `tf.optimizer.minimize` operations for updating
        the inverse model with gradient descent, and adds it to
        `self._training_ops` attribute.

        """

        next_states = tf.concat([self._placeholders['action_conditioned']['next_observations'],
                                 self._placeholders['action_free']['next_observations']], axis=0)
        
        prev_states = tf.concat([self._placeholders['action_conditioned']['observations'],
                                 self._placeholders['action_free']['observations']], axis=0)
    
        true_actions = self._placeholders['action_conditioned']['actions']
        action_con_obs = self._placeholders['action_conditioned']['observations']
        action_con_next_obs = self._placeholders['action_conditioned']['next_observations']
        action_free_obs = self._placeholders['action_free']['observations']
        action_free_next_obs = self._placeholders['action_free']['next_observations']

        if action_con_obs.shape[-1] == 6912 and not self._preprocessor_for_inverse:
            # 3 channel, 48x48 image
            action_con_obs = tf.reshape(action_con_obs, (-1, 48, 48, 3))
            action_con_next_obs = tf.reshape(action_con_next_obs, (-1, 48, 48, 3))
            action_free_obs = tf.reshape(action_free_obs, (-1, 48, 48, 3))
            action_free_next_obs = tf.reshape(action_free_next_obs, (-1, 48, 48, 3))
        combined_first_obs = tf.concat([action_con_obs, action_free_obs], axis=0)
        combined_next_obs = tf.concat([action_con_next_obs, action_free_next_obs], axis=0)
        combined_pred_actions = self._inverse_model([combined_first_obs, combined_next_obs])

        pred_seen_actions = combined_pred_actions[:256]
        pred_unseen_actions = combined_pred_actions[256:]


        inverse_model_loss = tf.compat.v1.losses.mean_squared_error(
                labels=true_actions, predictions=pred_seen_actions)


        if self._inverse_domain_shift:

            if self._paired_data_pool is not None:
                combined_paired_data = tf.concat([self._placeholders['paired_data']['obs_of_interaction'],
                                                  self._placeholders['paired_data']['obs_of_observation']], axis=0)
                paired_encodings = self._shared_preprocessor_model(combined_paired_data)
                interaction_encodings = paired_encodings[:256]
                observation_encodings = paired_encodings[256:]
                self._paired_loss = self._paired_loss_scale * tf.keras.losses.MeanSquaredError()(interaction_encodings, observation_encodings)
                
                self._paired_optimizer = tf.compat.v1.train.AdamOptimizer(
                        learning_rate=self._paired_loss_lr,
                        name='paired_loss_optimizer')
                paired_train_op = self._paired_optimizer.minimize(loss=self._paired_loss,
                                                                           var_list=self._shared_preprocessor_model.trainable_variables)
                self._training_ops.update({'paired_loss': paired_train_op})

            pred_domains = self._inverse_domain_model(prev_states)
            discriminator_loss = tf.keras.losses.BinaryCrossentropy()(self._domains_ph, pred_domains)
            generator_loss = tf.keras.losses.BinaryCrossentropy()(1.0 - self._domains_ph, pred_domains)

            self._inverse_model_ds_score = tf.reduce_sum(tf.cast(tf.abs(pred_domains - self._domains_ph) <= 0.5, tf.float32)) / 512

            self._inverse_model_ds_generator_loss = generator_loss
            self._inverse_model_ds_discriminator_loss = discriminator_loss


            inverse_model_loss = inverse_model_loss + generator_loss * self._inverse_model_domain_shift_generator_weight
            self._inverse_model_discriminator_loss = discriminator_loss * self._inverse_model_domain_shift_discriminator_weight

            self._inverse_discrim_optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._inverse_model_discrim_lr,
                name='inverse_model_discrim_optimizer')
            inverse_discrim_train_op = self._inverse_discrim_optimizer.minimize(loss=self._inverse_model_discriminator_loss,
                                                                                var_list=self._inverse_domain_model.trainable_variables)
            self._training_ops.update({'inverse_model_discriminator': inverse_discrim_train_op})


        self._inverse_model_optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._inverse_model_lr,
                name='inverse_model_optimizer')

        inverse_train_op = self._inverse_model_optimizer.minimize(loss=inverse_model_loss, var_list=self._inverse_model.trainable_variables)

        self._training_ops.update({'inverse_model': inverse_train_op})
        self._inverse_model_loss = inverse_model_loss

        self._observations_ph = prev_states
        self._next_observations_ph = next_states
        if not self._use_ground_truth_actions:
            self._actions_ph = tf.concat([true_actions, pred_unseen_actions], axis=0)
        else:
            print("\n\n\n\nUSING GROUND TRUTH ACTIONS\n\n\n\n\n\n")
            self._actions_ph = tf.concat([true_actions, self._placeholders['action_free']['actions']], axis=0)
        if self._use_zero_actions:
            self._actions_ph = tf.concat([true_actions, pred_unseen_actions * 0.0], axis=0)
        self._rewards_ph = tf.concat([self._placeholders['action_conditioned']['rewards'],
                                      self._placeholders['action_free']['rewards']], axis=0)
        self._terminals_ph = tf.concat([self._placeholders['action_conditioned']['terminals'],
                                        self._placeholders['action_free']['terminals']], axis=0)
        self._iteration_ph = self._placeholders['action_conditioned']['iteration']

    def _init_diagnostics_ops(self): 
        if not self._domain_shift:
            diagnosables = OrderedDict(( 
                ('Q_value', self._Q_values), 
                ('Q_loss', self._Q_losses), 
                ('policy_loss', self._policy_losses), 
                ('alpha', self._alpha),
                ('inverse_model_loss', self._inverse_model_loss),
            )) 
        else:
            diagnosables = OrderedDict((
                ('Q_value', self._Q_values),
                ('Q_loss', self._Q_losses),
                ('policy_loss', self._policy_losses),
                ('alpha', self._alpha),
                ('inverse_model_loss', self._inverse_model_loss),
                ('Q_domain_shift_losses', self._q_domain_losses),
                ('Q_raw_domain_shift_losses', self._q_raw_domain_losses),
                ('Q_discrim_domain_shift_losses', self._q_domain_discrim_losses),
                ('Q_domain_scores', self._q_domain_scores),
                ('policy_domain_shift_loss', self._policy_domain_loss),
            ))

        if self._inverse_domain_shift:
            diagnosables['inverse_model_domain_shift_discriminator'] = self._inverse_model_ds_discriminator_loss
            diagnosables['inverse_model_domain_shift_generator'] = self._inverse_model_ds_generator_loss
            diagnosables['inverse_model_domain_shift_score'] = self._inverse_model_ds_score

        if self._paired_data_pool is not None:
            diagnosables['paired_data_loss'] = self._paired_loss

        diagnostic_metrics = OrderedDict(( 
            ('mean', tf.reduce_mean), 
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)), 
        )) 
 
        self._diagnostics_ops = OrderedDict([ 
            (f'{key}-{metric_name}', metric_fn(values)) 
            for key, values in diagnosables.items() 
            for metric_name, metric_fn in diagnostic_metrics.items() 
        ]) 

