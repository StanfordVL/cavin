"""Model Predictive Control Agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.trajectories import trajectory
from tf_agents.utils import eager_utils

from networks.push import losses as push_losses
from networks.push import summaries as push_summaries
from policies import mpc_policy
from utils import loss_utils


nest = tf.contrib.framework.nest


MpcLossInfo = collections.namedtuple(
    'MpcLossInfo', ('encoding_loss', 'dynamics_loss'))


@gin.configurable
class MpcAgent(tf_agent.TFAgent):
    """Model Predictive Control Agent."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 # Policy.
                 policy_ctor=mpc_policy.MpcPolicy,
                 policy_config=None,
                 # Network.
                 dynamics_ctor=None,
                 state_encoder_ctor=None,
                 # Losses.
                 get_dynamics_weights=push_losses.get_dynamics_weights,
                 state_loss=push_losses.state_loss,
                 state_encoding_loss=push_losses.state_encoding_loss,
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
        """Creates a CAVIN Agent.
        """
        tf.Module.__init__(self, name=name)

        self._num_goal_steps = policy_config.NUM_GOAL_STEPS

        policy = policy_ctor(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            dynamics_ctor=dynamics_ctor,
            state_encoder_ctor=state_encoder_ctor,
            config=policy_config,
            is_training=True)

        self._dynamics = policy.dynamics
        self._state_encoder = policy.state_encoder

        self._get_dynamics_weights = get_dynamics_weights
        self._state_loss = state_loss
        self._state_encoding_loss = state_encoding_loss

        self._state_loss_weight = policy_config.STATE_LOSS_WEIGHT
        self._state_encoding_loss_weight = policy_config.ENCODING_LOSS_WEIGHT

        self._state_summary = state_summary
        self._delta_state_summary = delta_state_summary
        self._action_summary = action_summary
        self._delta_action_summary = delta_action_summary

        self._learning_rate = learning_rate
        self._gradient_clipping = gradient_clipping
        self._debug_summaries = debug_summaries
        self._summarize_grads_and_vars = summarize_grads_and_vars

        super(MpcAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy=policy,
            collect_policy=policy,
            train_sequence_length=self._num_goal_steps + 1,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter)

    def _experience_to_transitions(self, experience):
        transitions = trajectory.to_transition(experience)
        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action

        if self._num_goal_steps is None:
            assert (self.train_sequence_length is not None and
                    self.train_sequence_length == 2)
            # Sequence empty time dimension if critic network is stateless.
            time_steps, actions, next_time_steps = tf.nest.map_structure(
                    lambda t: tf.squeeze(t, axis=1),
                    (time_steps, actions, next_time_steps))

        return time_steps, actions, next_time_steps

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
                              tf.train.AdamOptimizer(self._learning_rate))

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
                              tf.train.AdamOptimizer(self._learning_rate))

        if self._debug_summaries:
            with tf.name_scope('Weights'):
                tf.compat.v2.summary.histogram(
                    name='dynamics_weights', data=dynamics_weights,
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

        self.train_step_counter.assign_add(1)

        total_loss = encoding_loss + dynamics_loss
        extra = MpcLossInfo(encoding_loss=encoding_loss,
                            dynamics_loss=dynamics_loss)
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

        if self._debug_summaries:
            with tf.name_scope('Weights'):
                tf.compat.v2.summary.histogram(
                    name='dynamics_weights', data=dynamics_weights,
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

        total_loss = encoding_loss + dynamics_loss
        extra = MpcLossInfo(encoding_loss=encoding_loss,
                            dynamics_loss=dynamics_loss)
        return tf_agent.LossInfo(loss=total_loss, extra=extra)

    def _apply_gradients(self, gradients, variables, optimizer):
        # list(...) is required for Python3.
        grads_and_vars = list(zip(gradients, variables))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(
                grads_and_vars, self._gradient_clipping)

        if self._summarize_grads_and_vars:
            eager_utils.add_variables_summaries(
                grads_and_vars, self.train_step_counter)
            eager_utils.add_gradients_summaries(
                grads_and_vars, self.train_step_counter)

        optimizer.apply_gradients(grads_and_vars)

    def _sample_gaussian_noise(self, means, stddevs):
        return means + stddevs * tf.random_normal(
            tf.shape(stddevs), 0., 1., dtype=tf.float32)

    def _normal_kld(self, z, z_mean, z_stddev, weights=1.0):
        kld_array = (loss_utils.log_normal(z, z_mean, z_stddev) -
                     loss_utils.log_normal(z, 0.0, 1.0))
        return tf.losses.compute_weighted_loss(kld_array, weights)

    def encoding_loss(self,
                      states,
                      actions,
                      next_states,
                      dynamics_weights):
        """Computes the encoding loss."""
        with tf.name_scope('loss_encoding'):

            state_loss = 0.0

            states_t = nest.map_structure(lambda x: x[:, 0], states)
            for t in range(self._num_goal_steps):
                actions_t = actions[:, t]
                next_states_t = nest.map_structure(
                    lambda x: x[:, t], next_states)

                # Prediction.
                pred_states_t = self._dynamics((states_t, actions_t))

                # Losses.
                state_loss += self._state_loss(
                    next_states_t,
                    pred_states_t,
                    dynamics_weights[:, t]) * self._state_loss_weight

                # Update.
                states_t = pred_states_t

            state_loss /= self._num_goal_steps

            tf.compat.v2.summary.scalar(
                name='state_loss',
                data=state_loss,
                step=self.train_step_counter)

        return state_loss

    def dynamics_loss(self,
                      states,
                      actions,
                      next_states,
                      dynamics_weights):
        """Computes the dynamics loss."""
        with tf.name_scope('loss_dynamics'):
            state_loss = 0.0
            state_encoding_loss = 0.0

            for t in range(self._num_goal_steps):
                states_t = nest.map_structure(
                    lambda x: x[:, t], states)
                actions_t = actions[:, t]
                next_states_t = nest.map_structure(
                    lambda x: x[:, t], next_states)

                # Prediction.
                pred_states_t = self._dynamics((states_t, actions_t))

                # Losses.
                state_loss += self._state_loss(
                    next_states_t,
                    pred_states_t,
                    dynamics_weights[:, t]) * self._state_loss_weight
                state_encoding_loss += self._state_encoding_loss(
                    next_states_t,
                    pred_states_t,
                    dynamics_weights[:, t]) * self._state_encoding_loss_weight

            state_loss /= self._num_goal_steps
            state_encoding_loss /= self._num_goal_steps

            tf.compat.v2.summary.scalar(
                name='state_loss',
                data=state_loss,
                step=self.train_step_counter)
            tf.compat.v2.summary.scalar(
                name='state_encoding_loss',
                data=state_encoding_loss,
                step=self.train_step_counter)

        return state_loss + state_encoding_loss
