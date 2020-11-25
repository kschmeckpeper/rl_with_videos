from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from scipy import ndimage

from .rl_algorithm import RLAlgorithm


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            plotter=None,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            store_extra_policy_info=False,
            domain_shift=False,
            domain_shift_weight=-0.01,
            domain_shift_weight_q=-0.01,
            stop_overtraining=False,
            train_policy_on_all_data=True,
            auxiliary_loss=False,
            auxiliary_loss_weight=0.01,

            should_augment=False,
            trans_dist=4,

            save_full_state=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = [q for q, _, _, _, _ in Qs]
        self._q_domain_models = [d for _, d, _, _, _ in Qs]
        self._q_auxiliary_domain_models = [d for _, _, d, _, _ in Qs]
        self._advantage_models = [d for _, _, _, d, _ in Qs]
        self._value_models = [d for _, _, _, _, d in Qs]
        print("q domain models:", self._q_domain_models)
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in self._Qs)


        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._Q_lr = lr
        self._q_discrim_lr = lr
        self._auxiliary_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize
        self._store_extra_policy_info = store_extra_policy_info

        self._save_full_state = save_full_state

        self._domain_shift = domain_shift
        self._domain_shift_weight = domain_shift_weight
        self._domain_shift_weight_q = domain_shift_weight_q
        self._domain_shift_weight_q_d = -domain_shift_weight_q
        self._stop_overtraining = stop_overtraining
        print("domain shift weight", domain_shift_weight)

        self._auxiliary_loss = auxiliary_loss
        self._auxiliary_loss_weight = auxiliary_loss_weight

        self._train_policy_on_all_data = train_policy_on_all_data

        self._should_augment = should_augment
        self._trans_dist = trans_dist

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._build()

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_augmentation()

        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_no_aug_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation_no_aug',
        )

        self._next_observations_no_aug_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation_no_aug',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

        if self._domain_shift:
            self._domains_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='domains',
            )

    def _init_augmentation(self):
        self._observations_ph = self._augment_image(self._observations_no_aug_ph)
        self._next_observations_ph = self._augment_image(self._next_observations_no_aug_ph)


    def _get_Q_target(self):
        next_actions, _ = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

#        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)


        print("q target:", Q_target)
        print("q values:", Q_values)
        Q_losses  = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        if self._domain_shift:
            pred_domains = [d([self._observations_ph]) for d in self._q_domain_models]
            pred_domains[0] = tf.Print(pred_domains[0], [tf.reduce_sum(self._domains_ph[:256])], "gt domains first half")
            pred_domains[0] = tf.Print(pred_domains[0], [tf.reduce_sum(self._domains_ph[256:])], "gt domains second half")
            for i in range(len(pred_domains)):
                pred_domains[i] = tf.Print(pred_domains[i], [tf.reduce_sum(pred_domains[i][:256])], "pred_domains {} first half".format(i))
                pred_domains[i] = tf.Print(pred_domains[i], [tf.reduce_sum(pred_domains[i][256:])], "pred_domains {} second half".format(i))
#            pred_domains[0] = tf.Print(pred_domains[0], [tf.reduce_sum(pred_domains[0][:256] < 0.5)], "first half less than 0.5")
#            pred_domains[0] = tf.Print(pred_domains[0], [tf.reduce_sum(self._domains_ph[:256] < 0.5)], "gt first half less than 0.5")
#            pred_domains[0] = tf.Print(pred_domains[0], [tf.reduce_sum(pred_domains[0][256:] < 0.5)], "second half less than 0.5")
#            pred_domains[0] = tf.Print(pred_domains[0], [tf.reduce_sum(self._domains_ph[256:] < 0.5)], "gt second half less than 0.5")
#            pred_domains[0] = tf.Print(pred_domains[0], [(tf.reduce_sum(tf.cast(self._domains_ph[:256] < 0.5, tf.float32)) + tf.reduce_sum(tf.cast(self._domains_ph[256:] >= 0.5, tf.float32))) / 512], "score for gt")
            self._q_domain_scores = tuple((tf.reduce_sum(tf.cast(pd[:256] < 0.5, tf.float32)) + tf.reduce_sum(tf.cast(pd[256:] >= 0.5, tf.float32))) / 512 for pd in pred_domains)
#            for i in range(len(pred_domains)):
#                pred_domains[0] = tf.Print(pred_domains[0], [self._q_domain_scores[i]], "domain_score {}".format(i))
            pred_domains = tuple(pred_domains)
#            domain_losses = [-tf.losses.sigmoid_cross_entropy(self._domains_ph, d) for d in pred_domains]
            domain_losses = [tf.keras.losses.BinaryCrossentropy()(self._domains_ph, d) for d in pred_domains]
            for i in range(len(domain_losses)):
                domain_losses[i] = tf.Print(domain_losses[i], [domain_losses[i]], "domain_losses {}".format(i))
            domain_losses = tuple(domain_losses)
            self._q_domain_losses = [d_loss * self._domain_shift_weight_q  for d_loss in domain_losses]
            self._q_domain_discrim_losses = [d_loss * self._domain_shift_weight_q_d for d_loss in domain_losses]
            if self._stop_overtraining:
                self._q_domain_losses = [d_loss * tf.cast(tf.stop_gradient(d_score) > 0.6, tf.float32) for (d_loss, d_score) in zip(self._q_domain_losses, self._q_domain_scores)]
                self._q_domain_discrim_losses = [d_loss * tf.cast(tf.stop_gradient(d_score) < 0.9, tf.float32) for (d_loss, d_score) in zip(self._q_domain_discrim_losses, self._q_domain_scores)]
            for i in range(len(domain_losses)):
                self._q_domain_losses[i] = tf.Print(self._q_domain_losses[i], [self._q_domain_losses[i]], "q domain losses" + str(i))
                self._q_domain_discrim_losses[i] = tf.Print(self._q_domain_discrim_losses[i], [self._q_domain_discrim_losses[i]], "q_domain_discrim" + str(i))
                self._q_domain_discrim_losses[i] = tf.Print(self._q_domain_discrim_losses[i], [self._q_domain_scores[i]], "q domain scores" + str(i))


            self._q_domain_losses = tuple(self._q_domain_losses)
            self._q_domain_discrim_losses = tuple(self._q_domain_discrim_losses)

            self._q_raw_domain_losses = domain_losses
            print("q domain losses:", domain_losses)
            Q_losses = tuple(q + domain for q, domain in zip(Q_losses, self._q_domain_losses))
#            Q_losses = tuple( domain for q, domain in zip(Q_losses, self._q_domain_losses))
            print("q_losses:", Q_losses)
            for n in self._q_domain_models:
                print("name:", n._name)

            self._q_discrim_optims = tuple(
                tf.train.AdamOptimizer(learning_rate=self._q_discrim_lr,
                    name="{}_{}_optimizer".format(qd._name, i))
                for i, qd in enumerate(self._q_domain_models))
            for qd in self._q_domain_models:
                print("q_discrim trainable:", qd.trainable_variables[4:])
            q_discrim_training_ops = tuple(
                qd_optimizer.minimize(loss=qd_loss, var_list=qd.trainable_variables[4:])
                for i, (qd, qd_loss, qd_optimizer)
                in enumerate(zip(self._q_domain_models, self._q_domain_discrim_losses, self._q_discrim_optims)))

            self._training_ops.update({'Q_d': tf.group(q_discrim_training_ops)})

            if self._auxiliary_loss:
                pred_ims = [d([self._observations_ph, self._actions_ph]) for d in self._q_auxiliary_domain_models]
                print("pred_ims:", pred_ims)
                pred_error = [tf.keras.losses.MeanSquaredError()(pred, self._next_observations_ph) for pred in pred_ims]

                
                self._auxiliary_pred_error = pred_error

                self._auxiliary_optims = tuple(
                    tf.train.AdamOptimizer(learning_rate=self._auxiliary_lr,
                        name="{}_{}_optimizer".format(model._name, i))
                    for i, model in enumerate(self._q_auxiliary_domain_models))

                auxiliary_training_ops = tuple(
                    am_optim.minimize(loss=pred_e * self._auxiliary_loss_weight, var_list=am.trainable_variables)
                    for i, (am, pred_e, am_optim)
                    in enumerate(zip(self._q_auxiliary_domain_models, pred_error, self._auxiliary_optims)))

                self._training_ops.update({'Aux_pred': tf.group(auxiliary_training_ops)})


        self._Q_losses = Q_losses
        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """
        print("domain shift?", self._domain_shift)
        print("observations:", self._observations_ph)
        actions, pred_domains = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

#        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

            

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

#        assert policy_kl_losses.shape.as_list() == [None, 1]

#        if self._domain_shift:
#            pred_domains = tf.math.sigmoid(pred_domains)
##            pred_domains = tf.Print(pred_domains, [tf.math.reduce_sum(self._domains_ph[:256])], "domains ph first half", summarize=512)
##            pred_domains = tf.Print(pred_domains, [tf.math.reduce_sum(self._domains_ph[256:])], "domains ph second half", summarize=512)
##            pred_domains = tf.Print(pred_domains, [pred_domains], "pred_domains_policy", summarize=512)
##            pred_domains = tf.Print(pred_domains, [tf.math.reduce_sum(pred_domains[:256])], "first half sum:")
##            pred_domains = tf.Print(pred_domains, [tf.math.reduce_sum(pred_domains[256:])], "second half sum")
#            domain_loss = - tf.losses.sigmoid_cross_entropy(self._domains_ph, pred_domains)
#            self._policy_domain_loss = domain_loss
#            print("policy domain loss:", domain_loss)
#            policy_kl_losses = policy_kl_losses + domain_loss * self._domain_shift_weight
#            print("combined policy loss:", policy_kl_losses)
        self._policy_domain_loss = 0.0

        self._policy_losses = policy_kl_losses
        if self._train_policy_on_all_data:
            policy_loss = tf.reduce_mean(policy_kl_losses)
        else:
            policy_loss = tf.reduce_mean(policy_kl_losses[:256])

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
            ('policy_loss', self._policy_losses),
            ('alpha', self._alpha)
        ))

        if self._domain_shift:
            diagnosables.update(('Q_domain_shift_losses', self._q_domain_losses))
            diagnosables.update(('policy_domain_shift_loss', self._policy_domain_loss))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
#        print("training ops:", self._training_ops)
        self._session.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _augment_image(self, flat_image):
        original_shape = flat_image.shape[1]
        if not self._should_augment:
            return flat_image
        image = tf.reshape(flat_image, (-1, 48, 48, 3))

        if self._should_augment:
            padding = tf.constant([[0, 0], [self._trans_dist, self._trans_dist], [self._trans_dist, self._trans_dist], [0, 0]])
            image = tf.pad(image, padding)
            image = tf.image.random_crop(image, (256, 48, 48, 3))

        flattened_image = tf.reshape(image, (-1, original_shape))
        return flattened_image

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_no_aug_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_no_aug_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._domain_shift:
            feed_dict[self._domains_ph] = np.zeros(batch['terminals'].shape)

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

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
        diagnostics = self._session.run(self._diagnostics_ops, feed_dict)

        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(batch['observations']).items()
        ]))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
