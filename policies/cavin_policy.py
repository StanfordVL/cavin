"""CAVIN policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step

from networks.push import networks as push_networks
from networks import samplers
from policies import mpc_policy
from policies.mpc_policy import expand_and_tile
from policies.mpc_policy import prune_plans


nest = tf.contrib.framework.nest


def get_state_dists(a, b):
    """Get the state distances.

    Args:
        a: The first state.
        b: The second state.

    Returns:
        The average distances.
    """
    assert ('position' in a) and ('position' in b), (
        'Debugging mode is only supported for PushEnv for now.')
    dists = tf.norm(
        a['position'][..., :2] - b['position'][..., :2],
        axis=-1)
    return tf.reduce_sum(dists, axis=1)


class CavinPolicy(mpc_policy.MpcPolicy):
    """CAVIN policy."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 dynamics_ctor=None,
                 meta_dynamics_ctor=None,
                 action_generator_ctor=None,
                 state_encoder_ctor=None,
                 effect_encoder_ctor=None,
                 get_reward=None,
                 get_state_dist=None,
                 config=None,
                 is_training=False):
        """Initialize."""
        self._debug = config.DEBUG and (not is_training)
        self._use_pruning = config.USE_PRUNING
        self._reward_thresh = config.REWARD_THRESH
        self._reward_thresh_high = config.REWARD_THRESH_HIGH
        self._num_steps = config.NUM_STEPS
        self._num_goal_steps = config.NUM_GOAL_STEPS
        self._num_sampled_cs = config.NUM_SAMPLED_CS
        self._num_sampled_zs = config.NUM_SAMPLED_ZS
        self._num_info_samples = config.NUM_INFO_SAMPLES
        assert self._num_steps % self._num_goal_steps == 0
        self._num_goals = int(self._num_steps / self._num_goal_steps)

        self._task_name = config.TASK_NAME
        self._layout_id = config.LAYOUT_ID

        self._c_spec = tensor_spec.TensorSpec(shape=[config.DIM_C], name='c')
        self._z_spec = tensor_spec.TensorSpec(shape=[config.DIM_Z], name='z')

        # Reward.
        self._get_reward = get_reward or push_networks.get_reward

        # Build.
        if len(action_spec.shape) == 1:
            step_action_spec = action_spec
        else:
            assert action_spec.shape[0] == self._num_goal_steps
            step_action_spec = tensor_spec.BoundedTensorSpec(
                shape=action_spec.shape[1:],
                dtype=action_spec.dtype,
                minimum=action_spec.minimum,
                maximum=action_spec.maximum,
                name=action_spec.name)

        self._c_sampler = samplers.NormalSampler(
            time_step_spec.observation,
            self._c_spec,
            num_steps=self._num_goals)
        self._z_sampler = samplers.NormalSampler(
            time_step_spec.observation,
            self._z_spec,
            num_steps=self._num_goals)

        state_encoder_ctor = state_encoder_ctor or push_networks.StateEncoder
        self._state_encoder = state_encoder_ctor(
            input_tensor_spec=time_step_spec.observation,
            config=config)
        self._state_encoder.create_variables()
        self._state_spec = self._state_encoder.output_tensor_spec

        dynamics_ctor = dynamics_ctor or push_networks.Dynamics
        self._dynamics = dynamics_ctor(
            input_tensor_spec=(self._state_spec, step_action_spec),
            config=config)
        self._dynamics.create_variables()

        effect_encoder_ctor = (effect_encoder_ctor or
                               push_networks.EffectEncoder)
        self._effect_encoder = effect_encoder_ctor(
            input_tensor_spec=(self._state_spec, self._c_spec),
            config=config)
        self._effect_encoder.create_variables()
        self._effect_spec = self._effect_encoder.output_tensor_spec

        meta_dynamics_ctor = meta_dynamics_ctor or push_networks.MetaDynamics
        self._meta_dynamics = meta_dynamics_ctor(
            input_tensor_spec=(self._state_spec, self._effect_spec),
            config=config)
        self._meta_dynamics.create_variables()

        action_generator_ctor = (action_generator_ctor or
                                 push_networks.ActionGenerator)
        self._action_generator = action_generator_ctor(
            input_tensor_spec=(
                self._state_spec, self._effect_spec, self._z_spec),
            config=config)
        self._action_generator.create_variables()

        if self._debug:
            assert 'position' in self._state_spec, (
                'Debugging mode is only supported for PushEnv for now.')
            position_shape = self._state_spec['position'].shape.as_list()
            action_shape = step_action_spec.shape.as_list()
            info_spec = {
                'pred_goals': tf.TensorSpec(
                    [self._num_info_samples, self._num_goals] + position_shape,
                    tf.float32, 'pred_goals'),
                'goal_terminations': tf.TensorSpec(
                    [self._num_info_samples, self._num_goals],
                    tf.bool, 'goal_terminations'),
                'actions': tf.TensorSpec(
                    [self._num_info_samples, self._num_steps] + action_shape,
                    tf.float32, 'actions'),
                'pred_states': tf.TensorSpec(
                    [self._num_info_samples, self._num_steps] + position_shape,
                    tf.float32, 'pred_states'),
                'terminations': tf.TensorSpec(
                    [self._num_info_samples, self._num_steps],
                    tf.bool, 'terminations'),
            }
        else:
            info_spec = ()

        super(mpc_policy.MpcPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=(),
            info_spec=info_spec,
            config=config)

    @property
    def c_spec(self):
        return self._c_spec

    @property
    def z_spec(self):
        return self._z_spec

    @property
    def effect_spec(self):
        return self._effect_spec

    @property
    def meta_dynamics(self):
        return self._meta_dynamics

    @property
    def action_generator(self):
        return self._action_generator

    @property
    def state_encoder(self):
        return self._state_encoder

    @property
    def effect_encoder(self):
        return self._effect_encoder

    def _variables(self):
        return (self._dynamics.variables +
                self._meta_dynamics.variables +
                self._action_generator.variables +
                self._state_encoder.variables +
                self._effect_encoder.variables)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')

    def _action(self, time_step, policy_state, seed):
        best_cs, info_high = self.plan_high(time_step, seed)
        assert best_cs.shape[0] == 1
        cs = tf.tile(best_cs, [self._num_sampled_zs, 1, 1])

        best_actions, info_low = self.plan_low(time_step, cs, seed)
        action = best_actions[:, :self._num_goal_steps, :]

        if self._debug:
            info_high.update(info_low)
            info = info_high
        else:
            info = ()

        return policy_step.PolicyStep(action, policy_state, info)

    def plan_high(self, time_step, seed):
        """High-level planning."""
        # Sample.
        sampled_cs = self._c_sampler(
            time_step.observation, self._num_sampled_cs, seed)

        # Predict.
        output_cs, pred_goals, rewards, terminations = self.predict_high(
            time_step, sampled_cs)

        # Select.
        cum_rewards = tf.reduce_sum(rewards, axis=1)
        _, best_inds = tf.nn.top_k(cum_rewards, k=1)
        best_cs = tf.gather(output_cs, best_inds, axis=0)

        # Info.
        if self._debug:
            assert 'position' in self._state_spec, (
                'Debugging mode is only supported for PushEnv for now.')
            info = {
                'pred_goals': pred_goals['position'],
                'goal_terminations': terminations,
            }
            _, info_inds = tf.nn.top_k(cum_rewards, k=self._num_info_samples)
            for key, value in info.items():
                # value = value[:self._num_info_samples]
                value = tf.gather(value, info_inds, axis=0)
                value = tf.expand_dims(value, axis=0)
                info[key] = value
        else:
            info = ()

        return best_cs, info

    def plan_low(self, time_step, cs, seed):
        """Low-level planning."""
        # Sample.
        sampled_zs = self._z_sampler(
            time_step.observation, self._num_sampled_zs, seed)

        # Predict.
        output_actions, pred_states, rewards, terminations = self.predict_low(
            time_step, cs, sampled_zs)

        # Select elites.
        cum_rewards = tf.reduce_sum(rewards, axis=1)
        _, best_inds = tf.nn.top_k(cum_rewards, k=1)
        best_actions = tf.gather(output_actions, best_inds, axis=0)

        # Info.
        if self._debug:
            assert 'position' in self._state_spec, (
                'Debugging mode is only supported for PushEnv for now.')
            info = {
                'actions': output_actions,
                'pred_states': pred_states['position'],
                'terminations': terminations,
            }
            _, info_inds = tf.nn.top_k(cum_rewards, k=self._num_info_samples)
            for key, value in info.items():
                # value = value[:self._num_info_samples]
                value = tf.gather(value, info_inds, axis=0)
                value = tf.expand_dims(value, axis=0)
                info[key] = value
        else:
            info = ()

        return best_actions, info

    def predict_high(self, time_step, sampled_cs):
        """High-level prediction.
        """
        num_samples = self._num_sampled_cs
        num_steps = self._num_goals
        assert sampled_cs.shape[0] == num_samples

        init_state = self._state_encoder(time_step.observation)
        states_t = nest.map_structure(
            lambda x: expand_and_tile(x, num_samples),
            nest.map_structure(lambda x: x[0], init_state))
        indices_t = tf.range(0, num_samples, dtype=tf.int64)

        cum_rewards = tf.zeros([num_samples], dtype=tf.float32)
        cum_terminations = tf.zeros([num_samples], dtype=tf.bool)

        pred_goals = []
        rewards = []
        terminations = []
        indices = []

        for t in range(num_steps):
            # Index the kept entries.
            if self._use_pruning:
                if t > 0:
                    states_t = nest.map_structure(
                        lambda x: tf.gather(x, indices_t), states_t)
                    cum_rewards = tf.gather(cum_rewards, indices_t)
                    cum_terminations = tf.gather(cum_terminations, indices_t)

            has_terminated = tf.identity(cum_terminations)

            # Index the samples.
            cs_t = sampled_cs[:, t]

            # Predict.
            effects_t = self._effect_encoder((states_t, cs_t))
            pred_goals_t = self._meta_dynamics((states_t, effects_t))
            rewards_t, terminations_t = self._get_reward(
                states=states_t,
                next_states=pred_goals_t,
                task_name=self._task_name,
                layout_id=self._layout_id,
                is_high_level=True)
            rewards_t = tf.where(
                has_terminated,
                tf.zeros_like(rewards_t),
                rewards_t)

            # Update reward and termintation.
            cum_rewards = tf.add(cum_rewards, rewards_t)
            cum_terminations = tf.logical_or(cum_terminations, terminations_t)

            # Prune.
            if self._use_pruning:
                if t == num_steps - 1:
                    indices_t = tf.range(0, num_samples, dtype=tf.int64)
                else:
                    indices_t = prune_plans(
                        cum_rewards,
                        cum_terminations,
                        has_terminated,
                        reward_thresh=self._reward_thresh_high)

            # Append lists.
            pred_goals.append(pred_goals_t)
            rewards.append(rewards_t)
            terminations.append(cum_terminations)
            indices.append(indices_t)

            # Update state.
            states_t = pred_goals_t

        # Gather selected plans.
        if self._use_pruning:
            # Recursively backtrack the index at each step.
            new_indices = [None] * num_steps
            new_indices[num_steps - 1] = tf.identity(
                indices[num_steps - 1])
            for t in range(num_steps - 2, -1, -1):
                new_indices[t] = tf.gather(
                    indices[t], new_indices[t + 1])

            # Index the kept entries at each step.
            cs = tf.gather(sampled_cs, new_indices[0], axis=0)

            for t in range(num_steps):
                pred_goals[t] = nest.map_structure(
                    lambda x: tf.gather(x, new_indices[t]),
                    pred_goals[t])
                rewards[t] = tf.gather(
                    rewards[t], new_indices[t])
                terminations[t] = tf.gather(
                    terminations[t], new_indices[t])
        else:
            cs = sampled_cs

        # Stack output tensors.
        pred_goals = tf.nest.map_structure(
            lambda *tensors: tf.stack(tensors, axis=1), *pred_goals)
        rewards = tf.stack(rewards, axis=1)
        terminations = tf.stack(terminations, axis=1)
        return cs, pred_goals, rewards, terminations

    def predict_low(self, time_step, cs, sampled_zs):
        """Low-level prediction."""
        num_samples = self._num_sampled_zs
        assert sampled_zs.shape[0] == num_samples

        init_state = self._state_encoder(time_step.observation)
        states_t = nest.map_structure(
            lambda x: expand_and_tile(x, num_samples),
            nest.map_structure(lambda x: x[0], init_state))

        actions = []
        pred_states = []
        rewards = []
        terminations = []

        # Index the samples.
        cs_t = cs[:, 0]
        zs_t = sampled_zs[:, 0]

        # Predict.
        effects_t = self._effect_encoder((states_t, cs_t))
        actions_t = self._action_generator((states_t, effects_t, zs_t))
        actions.append(actions_t)

        pred_states_t = states_t
        for k in range(self._num_goal_steps):
            actions_t_k = actions_t[:, k]
            pred_states_t = self._dynamics((pred_states_t, actions_t_k))
            pred_states.append(pred_states_t)
            terminations.append(tf.zeros([num_samples], dtype=tf.bool))

        pred_goals_t = self._meta_dynamics((states_t, effects_t))
        dists_t = get_state_dists(pred_goals_t, pred_states_t)
        rewards_t = tf.exp(-dists_t)
        rewards.append(rewards_t)

        # Stack output tensors.
        actions = tf.concat(actions, axis=1)  # Concat instead of stack.
        pred_states = tf.nest.map_structure(
            lambda *tensors: tf.stack(tensors, axis=1), *pred_states)
        rewards = tf.stack(rewards, axis=1)
        terminations = tf.stack(terminations, axis=1)
        return actions, pred_states, rewards, terminations
