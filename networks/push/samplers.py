"""Sampler classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf

from robovat.envs.push import heuristic_push_sampler

from networks import samplers


@gin.configurable
class HeuristicPushSampler(samplers.Base):
    """Samples starting pose using heuristics."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 num_steps=None,
                 config=None):
        self._output_tensor_spec = output_tensor_spec
        flat_output_tensor_spec = tf.nest.flatten(output_tensor_spec)
        if len(flat_output_tensor_spec) > 1:
            raise ValueError(
                'Only a single action is supported by this network.')

        self.num_steps = None

        self._sampler = heuristic_push_sampler.HeuristicPushSampler(
            cspace_low=config.ACTION.CSPACE.LOW,
            cspace_high=config.ACTION.CSPACE.HIGH,
            translation_x=config.ACTION.MOTION.TRANSLATION_X,
            translation_y=config.ACTION.MOTION.TRANSLATION_Y)

    def __call__(self, observation, num_samples, seed):
        position = tf.squeeze(observation['position'], 0)
        body_mask = tf.squeeze(observation['body_mask'], 0)
        num_episodes = tf.squeeze(observation['num_episodes'], 0)
        num_steps = tf.squeeze(observation['num_steps'], 0)
        action = tf.py_func(
            self._sampler.sample,
            [position, body_mask, num_episodes, num_steps, num_samples],
            [tf.float32])

        shape = list(self._output_tensor_spec.shape)
        action = tf.reshape(action, [num_samples] + shape)

        if self.num_steps is not None:
            action = tf.expand_dims(action, 0)

        return action
