"""Model predictive control policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_agents.specs import tensor_spec  # NOQA
from tf_agents.trajectories import policy_step

from networks.push import networks as push_networks
from networks import samplers
from policies import tf_policy


class MpcPolicy(tf_policy.TFPolicy):
    """Model predictive control policy."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 dynamics_ctor=None,
                 state_encoder_ctor=None,
                 get_reward=None,
                 config=None,
                 is_training=False):
        """Initialize."""
        self._debug = config.DEBUG and (not is_training)
        self._use_pruning = config.USE_PRUNING
        self._reward_thresh = config.REWARD_THRESH
        self._num_steps = config.NUM_STEPS
        self._num_goal_steps = config.NUM_GOAL_STEPS
        self._num_sampled_actions = config.NUM_SAMPLED_ACTIONS
        self._num_info_samples = config.NUM_INFO_SAMPLES
        assert self._num_steps % self._num_goal_steps == 0
        self._num_goals = int(self._num_steps / self._num_goal_steps)

        self._task_name = config.TASK_NAME
        self._layout_id = config.LAYOUT_ID

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

        self._action_sampler = samplers.ActionSampler(
            time_step_spec.observation,
            step_action_spec,
            num_steps=self._num_steps)

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

        # Info.
        if self._debug:
            assert 'position' in self._state_spec, (
                'Debugging mode is only supported for PushEnv for now.')
            position_shape = self._state_spec['position'].shape.as_list()
            action_shape = step_action_spec.shape.as_list()
            info_spec = {
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

        super(MpcPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=(),
            info_spec=info_spec,
            config=config)

    @property
    def state_spec(self):
        return self._state_spec

    @property
    def dynamics(self):
        return self._dynamics

    @property
    def state_encoder(self):
        return self._state_encoder

    def _variables(self):
        return self._state_encoder.variables + self._dynamics.variables

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')

    def _action(self, time_step, policy_state, seed):
        best_actions, info = self.plan(time_step, seed)
        action = best_actions[:, :self._num_goal_steps, :]
        return policy_step.PolicyStep(action, policy_state, info)

    def plan(self, time_step, seed):
        """High-level planning."""
        # Sample.
        sampled_actions = self._action_sampler(
            time_step.observation,
            self._num_sampled_actions,
            seed)

        # Predict.
        output_actions, pred_states, rewards, terminations = self.predict(
            time_step, sampled_actions)

        # Select.
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

    def predict(self, time_step, sampled_actions):
        """Prediction."""
        num_samples = self._num_sampled_actions
        num_steps = self._num_goals
        assert sampled_actions.shape[0] == num_samples

        init_state = self._state_encoder(time_step.observation)
        states_t = tf.nest.map_structure(
            lambda x: expand_and_tile(x, num_samples),
            tf.nest.map_structure(lambda x: x[0], init_state))
        indices_t = tf.range(0, num_samples, dtype=tf.int64)

        cum_rewards = tf.zeros([num_samples], dtype=tf.float32)
        cum_terminations = tf.zeros([num_samples], dtype=tf.bool)

        actions = []
        pred_states = []
        rewards = []
        terminations = []
        indices = []

        for t in range(num_steps):
            # Index the kept entries.
            if self._use_pruning:
                if t > 0:
                    states_t = tf.nest.map_structure(
                        lambda x: tf.gather(x, indices_t), states_t)
                    cum_rewards = tf.gather(cum_rewards, indices_t)
                    cum_terminations = tf.gather(cum_terminations, indices_t)

            has_terminated = tf.identity(cum_terminations)

            # Index.
            actions_t = sampled_actions[:, t * self._num_goal_steps:
                                        (t + 1) * self._num_goal_steps]

            # Predict.
            _pred_states_t = []
            pred_states_t_k = states_t
            for k in range(self._num_goal_steps):
                actions_t_k = actions_t[:, k]
                pred_states_t_k = self._dynamics(
                    (pred_states_t_k, actions_t_k))
                _pred_states_t.append(pred_states_t_k)
            _pred_states_t = tf.nest.map_structure(
                lambda *x: tf.stack(x, axis=1),
                *_pred_states_t)
            pred_states_t = pred_states_t_k

            rewards_t, terminations_t = self._get_reward(
                states=states_t,
                next_states=pred_states_t,
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
                indices_t = prune_plans(
                    cum_rewards,
                    cum_terminations,
                    has_terminated,
                    reward_thresh=self._reward_thresh)

            _cum_terminations_t = (
                [has_terminated] * (self._num_goal_steps - 1)
                + [cum_terminations])

            # Append lists.
            actions.append(actions_t)
            pred_states.append(_pred_states_t)
            rewards.append(rewards_t)
            terminations.append(_cum_terminations_t)
            indices.append(indices_t)

            # Update state.
            states_t = pred_states_t

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
            for t in range(num_steps):
                actions[t] = tf.gather(
                    actions[t], new_indices[t])
                pred_states[t] = tf.nest.map_structure(
                    lambda x: tf.gather(x, new_indices[t]),
                    pred_states[t])
                rewards[t] = tf.gather(
                    rewards[t], new_indices[t])
                terminations[t] = tf.gather(
                    terminations[t], new_indices[t])

        # Stack output tensors.
        actions = tf.concat(actions, axis=1)  # Concat instead of stack.
        pred_states = tf.nest.map_structure(
            lambda *tensors: tf.concat(tensors, axis=1), *pred_states)
        rewards = tf.stack(rewards, axis=1)
        terminations = tf.concat(terminations, axis=1)
        return actions, pred_states, rewards, terminations


def expand_and_tile(x, multiple, axis=0):
    """Expand and tile a selected axis.

    Args:
        x: The input tensor.
        multiple: Number of replicas to tile.
        axis: The axis to be expanded and tiled.

    Return:
        The output tensor.
    """
    n_dims = len(x.shape)
    multiples = axis * [1] + [multiple] + (n_dims - axis) * [1]
    return tf.tile(tf.expand_dims(x, axis), multiples)


def prune_plans(rewards, terminations, has_terminated, reward_thresh=0.0):
    """Prune plans.

    Args:
        rewards: Reward of the current step of shape [batch_size].
        terminations: Termination of the current step of shape [batch_size].
        has_terminated: Tensor of shape [batch_size] which indicates if the
            plan has already terminated before the current step.
        reward_thresh: The threshold for the reward.

    Returns:
        a
    """
    batch_size = rewards.shape[0]
    assert batch_size == terminations.shape[0]
    assert batch_size == has_terminated.shape[0]

    terminations_this_step = tf.logical_and(
        terminations,
        tf.logical_not(has_terminated))

    # Remove a plan if it terminates at this step with a reward less the
    # threshold. Keep a plan if it is not removed (either it is still alive,
    # or it terminates at this step with a descent reward, or it has already
    # terminated). Grow a plan if it is still alive.
    remove_masks = tf.logical_and(
        terminations_this_step,
        tf.less(rewards, reward_thresh))
    keep_masks = tf.logical_not(remove_masks)
    grow_masks = tf.logical_and(keep_masks, tf.logical_not(has_terminated))

    # Get the plan indices for each category.
    inds = tf.range(0, batch_size, dtype=tf.int64)
    remove_inds = tf.boolean_mask(inds, remove_masks)
    keep_inds = tf.boolean_mask(inds, keep_masks)
    grow_inds = tf.boolean_mask(inds, grow_masks)

    # Get the number of plans for each category.
    num_remove = tf.reduce_sum(tf.cast(remove_masks, tf.int64))
    num_grow = tf.reduce_sum(tf.cast(grow_masks, tf.int64))

    # Sample
    def sample_replace_inds():
        samples = tf.random.uniform([num_remove], 0, num_grow, tf.int64)
        return tf.gather(grow_inds, samples)

    replace_inds = tf.cond(
        tf.greater(num_grow, 0),
        lambda: sample_replace_inds(),
        lambda: remove_inds)

    # Return kept and replaced indices.
    ret_inds = tf.concat([keep_inds, replace_inds], axis=0)
    ret_inds = tf.reshape(ret_inds, [batch_size])
    return ret_inds
