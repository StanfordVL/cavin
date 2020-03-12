"""Cascaded Variational Inference Agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec  # NOQA

from tf_agents.agents import tf_agent

from agents import mpc_agent
from networks.push import networks as push_networks
from networks.push import summaries as push_summaries
from networks.push import losses as push_losses
from policies import cavin_policy


CavinLossInfo = collections.namedtuple(
    'CavinLossInfo', ('encoding_loss', 'dynamics_loss', 'vae_loss'))


@gin.configurable
class CavinAgent(mpc_agent.MpcAgent):
    """Cascaded Variational Inference Agent."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 # Policy.
                 policy_ctor=cavin_policy.CavinPolicy,
                 policy_config=None,
                 # Network.
                 dynamics_ctor=None,
                 meta_dynamics_ctor=None,
                 action_generator_ctor=None,
                 c_inference_network_ctor=None,
                 z_inference_network_ctor=None,
                 state_encoder_ctor=None,
                 effect_encoder_ctor=None,
                 # Losses.
                 get_dynamics_weights=push_losses.get_dynamics_weights,
                 get_c_weights=push_losses.get_c_weights,
                 get_z_weights=push_losses.get_z_weights,
                 state_loss=push_losses.state_loss,
                 state_encoding_loss=push_losses.state_encoding_loss,
                 action_loss=push_losses.action_loss,
                 # Summaries.
                 state_summary=push_summaries.state_summary,
                 delta_state_summary=push_summaries.delta_state_summary,
                 action_summary=push_summaries.action_summary,
                 delta_action_summary=push_summaries.delta_action_summary,
                 # Policy and Network.
                 learning_rate=0.0001,
                 gradient_clipping=None,
                 # Misc.
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 train_step_counter=None,
                 name=None):
        """Creates a CAVIN Agent."""
        tf.Module.__init__(self, name=name)

        self._num_goal_steps = policy_config.NUM_GOAL_STEPS

        self._state_loss_weight = policy_config.STATE_LOSS_WEIGHT
        self._state_encoding_loss_weight = policy_config.ENCODING_LOSS_WEIGHT
        self._action_loss_weight = policy_config.ACTION_LOSS_WEIGHT
        self._kld_loss_weight = policy_config.KLD_LOSS_WEIGHT
        self._cfr_loss_weight = policy_config.CFR_LOSS_WEIGHT

        self._use_cfr = policy_config.USE_CFR

        policy = policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            dynamics_ctor=dynamics_ctor,
            meta_dynamics_ctor=meta_dynamics_ctor,
            action_generator_ctor=action_generator_ctor,
            state_encoder_ctor=state_encoder_ctor,
            effect_encoder_ctor=effect_encoder_ctor,
            config=policy_config,
            is_training=True)

        pred_action_spec = tensor_spec.BoundedTensorSpec(
            shape=[self._num_goal_steps] + action_spec.shape.as_list(),
            dtype=action_spec.dtype,
            minimum=action_spec.minimum,
            maximum=action_spec.maximum,
            name=action_spec.name)

        self._dynamics = policy.dynamics
        self._meta_dynamics = policy.meta_dynamics
        self._action_generator = policy.action_generator
        self._state_encoder = policy.state_encoder
        self._effect_encoder = policy.effect_encoder

        c_inference_network_ctor = (c_inference_network_ctor or
                                    push_networks.CInferenceNetwork)
        self._c_inference_network = c_inference_network_ctor(
            input_tensor_spec=(policy.state_spec, policy.state_spec),
            config=policy_config)
        self._c_inference_network.create_variables()

        z_inference_network_ctor = (z_inference_network_ctor or
                                    push_networks.ZInferenceNetwork)
        self._z_inference_network = z_inference_network_ctor(
            input_tensor_spec=(
                policy.state_spec, pred_action_spec, policy.effect_spec),
            config=policy_config)
        self._z_inference_network.create_variables()

        self._get_dynamics_weights = get_dynamics_weights
        self._get_c_weights = get_c_weights
        self._get_z_weights = get_z_weights
        self._state_loss = state_loss
        self._state_encoding_loss = state_encoding_loss
        self._action_loss = action_loss

        self._state_summary = state_summary
        self._delta_state_summary = delta_state_summary
        self._action_summary = action_summary
        self._delta_action_summary = delta_action_summary

        self._learning_rate = learning_rate
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars

        self._encoding_optimizer = tf.compat.v1.train.AdamOptimizer(
            self._learning_rate)
        self._dynamics_optimizer = tf.compat.v1.train.AdamOptimizer(
            self._learning_rate)
        self._vae_optimizer = tf.compat.v1.train.AdamOptimizer(
            self._learning_rate)

        super(mpc_agent.MpcAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=self._num_goal_steps + 1,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

    def _train(self, experience, weights):
        """Returns a train op to update the agent's networks.

        This method trains with the provided batched experience.

        Args:
            experience: A time-stacked trajectory object.
            weights: Optional scalar or elementwise (per-batch-entry)
                importance weights.

        Returns:
            A train_op.

        Raises:
            ValueError: If optimizers are None and no default value was
                provided to the constructor.
        """
        time_steps, actions, next_time_steps = self._experience_to_transitions(
            experience)
        tf.nest.assert_same_structure(actions, self.action_spec)
        tf.nest.assert_same_structure(time_steps, self.time_step_spec)
        tf.nest.assert_same_structure(next_time_steps, self.time_step_spec)

        dynamics_weights = self._get_dynamics_weights(
            time_steps, actions, next_time_steps)
        c_weights = self._get_c_weights(
            time_steps, actions, next_time_steps)
        z_weights = self._get_z_weights(
            time_steps, actions, next_time_steps)

        encoding_variables = self._state_encoder.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert encoding_variables, (
                'No trainable encoding variables to optimize.')
            tape.watch(encoding_variables)
            states = self._state_encoder(time_steps.observation)
            next_states = self._state_encoder(next_time_steps.observation)
            encoding_loss = self.encoding_loss(
                states,
                actions,
                next_states,
                dynamics_weights=dynamics_weights)
        tf.debugging.check_numerics(
            encoding_loss, 'encoding loss is inf or nan.')
        encoding_grads = tape.gradient(encoding_loss, encoding_variables)
        self._apply_gradients(encoding_grads,
                              encoding_variables,
                              self._encoding_optimizer)

        dynamics_variables = self._dynamics.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert dynamics_variables, (
                'No trainable dynamics variables to optimize.')
            tape.watch(dynamics_variables)
            dynamics_loss = self.dynamics_loss(
                states,
                actions,
                next_states,
                dynamics_weights=dynamics_weights)
        tf.debugging.check_numerics(
            dynamics_loss, 'dynamics loss is inf or nan.')
        dynamics_grads = tape.gradient(dynamics_loss, dynamics_variables)
        self._apply_gradients(dynamics_grads,
                              dynamics_variables,
                              self._dynamics_optimizer)

        vae_variables = (self._c_inference_network.trainable_variables +
                         self._z_inference_network.trainable_variables +
                         self._effect_encoder.trainable_variables +
                         self._meta_dynamics.trainable_variables +
                         self._action_generator.trainable_variables)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert vae_variables, (
                'No trainable vae variables to optimize.')
            tape.watch(vae_variables)
            vae_loss = self.vae_loss(
                states,
                actions,
                next_states,
                c_weights=c_weights,
                z_weights=z_weights)
        tf.debugging.check_numerics(vae_loss, 'vae loss is inf or nan.')
        vae_grads = tape.gradient(vae_loss, vae_variables)
        self._apply_gradients(vae_grads,
                              vae_variables,
                              self._vae_optimizer)

        if self._debug_summaries:
            with tf.name_scope('Weights'):
                tf.compat.v2.summary.histogram(
                    name='dynamics_weights', data=dynamics_weights,
                    step=self.train_step_counter)
                tf.compat.v2.summary.histogram(
                    name='c_weights', data=c_weights,
                    step=self.train_step_counter)
                tf.compat.v2.summary.histogram(
                    name='z_weights', data=z_weights,
                    step=self.train_step_counter)

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='encoding_loss',
                data=encoding_loss,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='dynamics_loss',
                data=dynamics_loss,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='vae_loss',
                data=vae_loss,
                step=self.train_step_counter)

        self.train_step_counter.assign_add(1)

        total_loss = encoding_loss + dynamics_loss + vae_loss
        extra = CavinLossInfo(encoding_loss=encoding_loss,
                              dynamics_loss=dynamics_loss,
                              vae_loss=vae_loss)
        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def eval(self, experience, weights=None):
        """Returns a eval op to update the agent's networks.

        Args:
            experience: A time-stacked trajectory object.
            weights: Optional scalar or elementwise (per-batch-entry)
                importance weights.

        Returns:
            A eval_op.
        """
        time_steps, actions, next_time_steps = self._experience_to_transitions(
            experience)
        tf.nest.assert_same_structure(actions, self.action_spec)
        tf.nest.assert_same_structure(time_steps, self.time_step_spec)
        tf.nest.assert_same_structure(next_time_steps, self.time_step_spec)

        dynamics_weights = self._get_dynamics_weights(
            time_steps, actions, next_time_steps)
        c_weights = self._get_c_weights(
            time_steps, actions, next_time_steps)
        z_weights = self._get_z_weights(
            time_steps, actions, next_time_steps)

        states = self._state_encoder(time_steps.observation)
        next_states = self._state_encoder(next_time_steps.observation)
        encoding_loss = self.encoding_loss(
            states,
            actions,
            next_states,
            dynamics_weights=dynamics_weights)
        dynamics_loss = self.dynamics_loss(
            states,
            actions,
            next_states,
            dynamics_weights=dynamics_weights)
        vae_loss = self.vae_loss(
            states,
            actions,
            next_states,
            c_weights=c_weights,
            z_weights=z_weights)

        if self._debug_summaries:
            with tf.name_scope('Weights'):
                tf.compat.v2.summary.histogram(
                    name='dynamics_weights', data=dynamics_weights,
                    step=self.train_step_counter)
                tf.compat.v2.summary.histogram(
                    name='c_weights', data=c_weights,
                    step=self.train_step_counter)
                tf.compat.v2.summary.histogram(
                    name='z_weights', data=z_weights,
                    step=self.train_step_counter)

        with tf.name_scope('Losses'):
            tf.compat.v2.summary.scalar(
                name='encoding_loss',
                data=encoding_loss,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='dynamics_loss',
                data=dynamics_loss,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='vae_loss',
                data=vae_loss,
                step=self.train_step_counter)

        total_loss = encoding_loss + dynamics_loss + vae_loss
        extra = CavinLossInfo(encoding_loss=encoding_loss,
                              dynamics_loss=dynamics_loss,
                              vae_loss=vae_loss)
        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def vae_loss(self,
                 states,
                 actions,
                 next_states,
                 c_weights,
                 z_weights):
        """Computes the variational inference loss."""
        with tf.name_scope('loss_vae'):
            init_states = tf.nest.map_structure(lambda x: x[:, 0], states)
            next_goals = tf.nest.map_structure(lambda x: x[:, -1], next_states)

            ##################
            # High-level inference and prediction.
            ##################
            # Inference.
            c_means, c_stddevs = self._c_inference_network(
                (init_states, next_goals))
            cs = self._sample_gaussian_noise(c_means, c_stddevs)
            effects = self._effect_encoder((init_states, cs))
            pred_goals = self._meta_dynamics((init_states, effects))

            # Losses
            c_kld = self._normal_kld(
                cs,
                c_means,
                c_stddevs,
                c_weights) * self._kld_loss_weight
            goal_loss = self._state_loss(
                next_goals,
                pred_goals,
                c_weights) * self._state_loss_weight
            goal_encoding_loss = self._state_encoding_loss(
                next_goals,
                pred_goals,
                c_weights) * self._state_encoding_loss_weight

            # Summaries.
            tf.compat.v2.summary.scalar(
                name='c_kld', data=c_kld,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='goal_loss', data=goal_loss,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='goal_encoding_loss', data=goal_encoding_loss,
                step=self.train_step_counter)

            if self._debug_summaries:
                self._state_summary(
                    'goals_next', next_goals,
                    step=self.train_step_counter)
                self._delta_state_summary(
                    'goals_next', init_states, next_goals,
                    step=self.train_step_counter)
                self._state_summary(
                    'goals_pred', pred_goals,
                    step=self.train_step_counter)
                self._delta_state_summary(
                    'goals_pred', init_states, pred_goals,
                    step=self.train_step_counter)
                self._delta_state_summary(
                    'goals_delta', next_goals, pred_goals,
                    step=self.train_step_counter)

                tf.compat.v2.summary.histogram(
                    name='cs', data=cs,
                    step=self.train_step_counter)
                tf.compat.v2.summary.histogram(
                    name='c_means', data=c_means,
                    step=self.train_step_counter)
                tf.compat.v2.summary.histogram(
                    name='c_stddevs', data=c_stddevs,
                    step=self.train_step_counter)

            ##################
            # Low-level inference and prediction.
            ##################
            # Inference.
            z_means, z_stddevs = self._z_inference_network(
                (init_states, actions, effects))
            zs = self._sample_gaussian_noise(z_means, z_stddevs)

            # Predict
            pred_actions = self._action_generator((init_states, effects, zs))

            # Losses
            z_kld = self._normal_kld(
                zs,
                z_means,
                z_stddevs,
                c_weights) * self._kld_loss_weight

            action_loss = 0.0
            for t in range(self._num_goal_steps):
                action_loss += self._action_loss(
                    actions[:, t],
                    pred_actions[:, t],
                    c_weights) * self._action_loss_weight
            action_loss /= self._num_goal_steps

            # Summaries.
            tf.compat.v2.summary.scalar(
                name='z_kld', data=z_kld,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='action_loss', data=action_loss,
                step=self.train_step_counter)

            # Counterfactual regularization.
            if self._use_cfr:
                cfr_states_t = init_states
                for t in range(self._num_goal_steps):
                    cfr_states_t = self._dynamics(
                        (cfr_states_t, pred_actions[:, t]))

                cfr_loss = self._state_loss(
                    next_goals,
                    cfr_states_t,
                    c_weights) * self._state_loss_weight
                cfr_loss *= self._cfr_loss_weight

                # Summaries.
                tf.compat.v2.summary.scalar(
                    name='cfr_loss',
                    data=cfr_loss,
                    step=self.train_step_counter)
            else:
                cfr_loss = 0.0

        return (c_kld + goal_loss + goal_encoding_loss + z_kld + action_loss +
                cfr_loss)
