"""Sampler classes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow as tf
from tf_agents.specs import tensor_spec  # NOQA


class Base(object):
    """Base calss of samplers."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 num_steps=None,
                 config=None):
        self._output_tensor_spec = output_tensor_spec

    def __call__(self, observation, num_samples, seed):
        raise NotImplementedError


@gin.configurable
class ActionSampler(Base):
    """Uniformly samples action."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 num_steps=None,
                 config=None):
        self._output_tensor_spec = output_tensor_spec
        self._num_steps = num_steps

    def __call__(self, observation, num_samples, seed):
        if self._num_steps is None:
            outer_dims = [num_samples]
        else:
            outer_dims = [num_samples, self._num_steps]

        return tensor_spec.sample_spec_nest(
            self._output_tensor_spec, seed=seed, outer_dims=outer_dims)


@gin.configurable
class NormalSampler(Base):
    """Uniformly samples latent action."""

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 num_steps=None,
                 config=None):
        self._output_tensor_spec = output_tensor_spec
        self._num_steps = num_steps

        assert len(output_tensor_spec.shape) == 1
        self._dim = int(output_tensor_spec.shape[-1])

    def __call__(self, observation, num_samples, seed):
        if self._num_steps is None:
            outer_dims = [num_samples]
        else:
            outer_dims = [num_samples, self._num_steps]

        shape = outer_dims + [self._dim]
        z = tf.random_normal(shape, 0.0, 1.0, dtype=tf.float32)

        return z
