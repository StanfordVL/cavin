"""Random policies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step

from policies import tf_policy

nest = tf.contrib.framework.nest


class RandomPolicy(random_tf_policy.RandomTFPolicy):
    """Sample random antipodal grasps."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):

        super(RandomPolicy, self).__init__(
            time_step_spec,
            action_spec)


class RandomPrimitivePolicy(tf_policy.TFPolicy):
    """Random policy using modes"""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 config=None):
        self._debug = config.DEBUG
        self._primitives = np.array(config.PRIMITIVES, dtype=np.float32)
        self._num_modes = self._primitives.shape[0]
        self._noise = np.array(config.NOISE, dtype=np.float32)
        self._use_uniform_action = config.USE_UNIFORM_ACTION

        super(RandomPrimitivePolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            config=config,)

    def _action(self, time_step, policy_state, seed):
        if self._use_uniform_action:
            use_mode = tf.random.categorical(
                [[1., 1.]],
                1,
                dtype=tf.int64)
            use_mode = tf.squeeze(use_mode, 0)
        else:
            use_mode = tf.cast([1], tf.int64)

        mode = tf.random.uniform(
            [1],
            minval=0,
            maxval=self._num_modes,
            dtype=tf.int64)

        # Sample action by mode.
        noise = tf.random.uniform(
            self.action_spec()['action'].shape,
            minval=-1.0,
            maxval=1.0,
            dtype=tf.float32) * self._noise
        mode_action = tf.gather(
            self._primitives,
            mode)
        mode_action += tf.expand_dims(noise, 0)

        # TODO(kuanfang): Assume actions are between [-1, 1].
        mode_action = tf.clip_by_value(mode_action, -1, 1)

        # Sample action uniformly from the action space.
        uniform_action = tensor_spec.sample_spec_nest(
            self.action_spec()['action'], seed=seed, outer_dims=[1])

        # Choose the action between the two.
        action = tf.where(tf.cast(use_mode, tf.bool),
                          mode_action,
                          uniform_action)

        # Debug
        if self._debug:
            print_op = tf.print(
                'use_mode: ', use_mode, '\n',
                'mode: ', mode, '\n',
                'noise: ', noise, '\n',
                'mode_action: ', mode_action, '\n',
                'uniform_action: ', uniform_action, '\n',
                'action: ', action, '\n',
            )
            with tf.control_dependencies([print_op]):
                action = tf.identity(action)

        action = {
            'action': action,
            'use_mode': use_mode,
            'mode': mode,
        }

        return policy_step.PolicyStep(action, policy_state)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')
