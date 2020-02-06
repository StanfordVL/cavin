"""Base extension to network to simplify copy operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
# import sys
import six
import tensorflow as tf

from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step

from tensorflow.python.training.tracking import base    # TF internal
from tensorflow.python.util import tf_decorator    # TF internal
from tensorflow.python.util import tf_inspect    # TF internal


framework = tf.contrib.framework
nest = tf.contrib.framework.nest


class _NetworkMeta(abc.ABCMeta):
    """Meta class for Network object.

    We mainly use this class to capture all args to `__init__` of all `Network`
    instances, and store them in `instance._saved_kwargs`.    This in turn is
    used by the `instance.copy` method.
    """

    def __new__(mcs, classname, baseclasses, attrs):
        """Control the creation of subclasses of the Network class.

        Args:
            classname: The name of the subclass being created.
            baseclasses: A tuple of parent classes.
            attrs: A dict mapping new attributes to their values.

        Returns:
            The class object.

        Raises:
            RuntimeError: if the class __init__ has *args in its signature.
        """
        if baseclasses[0] == object:
            # This is just Network below.    Return early.
            return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)

        init = attrs.get("__init__", None)

        if not init:
            # This wrapper class does not define an __init__.    When someone
            # creates the object, the __init__ of its parent class will be
            # called.    We will call that __init__ instead separately since the
            # parent class is also a subclass of Network.    Here just create
            # the class and return.
            return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)

        arg_spec = tf_inspect.getargspec(init)
        if arg_spec.varargs is not None:
            raise RuntimeError(
                    '%s.__init__ function accepts *args.'
                    'This is not allowed.' %
                    classname)

        def _capture_init(self, *args, **kwargs):
            """Captures init args and kwargs into `_saved_kwargs`."""
            if len(args) > len(arg_spec.args) + 1:
                # Error case: more inputs than args.    Call init so that the
                # appropriate error can be raised to the user.
                init(self, *args, **kwargs)
            for i, arg in enumerate(args):
                # Add +1 to skip `self` in arg_spec.args.
                kwargs[arg_spec.args[1 + i]] = arg
            init(self, **kwargs)
            # Avoid auto tracking which prevents keras from tracking layers that
            # are passed as kwargs to the Network.
            with base.no_automatic_dependency_tracking_scope(self):
                setattr(self, "_saved_kwargs", kwargs)

        attrs["__init__"] = tf_decorator.make_decorator(init, _capture_init)
        return abc.ABCMeta.__new__(mcs, classname, baseclasses, attrs)


@six.add_metaclass(_NetworkMeta)
class Network(object):
    """Base extension to network to simplify copy operations."""

    def __init__(self, input_tensor_spec, state_spec, name, mask_split_fn=None):
        """Creates an instance of `Network`.

        Args:
            input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing
                the input observations.
            state_spec: A nest of `tensor_spec.TensorSpec` representing the
                state needed by the network. Use () if none.
            name: A string representing the name of the network.
                mask_split_fn: A function used for masking valid/invalid actions
                with each state of the environment. The function takes in a full
                observation and returns a tuple consisting of 1) the part of the
                observation intended as input to the network and 2) the mask. An
                example mask_split_fn could be as simple as:
                    ```
                    def mask_split_fn(observation):
                        return observation['network_input'], observation['mask']
                    ```
                    If None, masking is not applied.
        """
        self._name = name
        self._input_tensor_spec = input_tensor_spec
        self._output_tensor_spec = None
        self._state_spec = state_spec
        self._mask_split_fn = mask_split_fn

        self._built = False

    @property
    def name(self):
        return self._name

    @property
    def state_spec(self):
        return self._state_spec

    @property
    def built(self):
        return self._built

    @property
    def weights(self):
        return self._weights

    @property
    def trainable_weights(self):
        return self._trainable_weights

    @property
    def non_trainable_weights(self):
        return self._non_trainable_weights

    def create_variables(self):
        if not self.built:
            random_input = tensor_spec.sample_spec_nest(
                self.input_tensor_spec, outer_dims=(1,))
            step_type = tf.expand_dims(time_step.StepType.FIRST, 0)
            output_tensors = self.__call__(random_input, step_type, None)

            with tf.variable_scope(self._name):
                scope = tf.get_variable_scope()
                self._weights = framework.get_variables(
                        scope=scope)
                self._trainable_weights = framework.get_trainable_variables(
                        scope=scope)
                self._non_trainable_weights = [
                    var for var in self._weights
                    if var not in self._trainable_weights]

            if self._output_tensor_spec is None:
                self._output_tensor_spec = nest.map_structure(
                    lambda t: tensor_spec.TensorSpec.from_tensor(
                        tf.squeeze(t, axis=0), name=t.name),
                    output_tensors)

    @property
    def input_tensor_spec(self):
        """Returns the spec of the input to the network of type InputSpec."""
        return self._input_tensor_spec

    @property
    def output_tensor_spec(self):
        """Returns the spec of the input to the network of type OutputSpec."""
        assert self._output_tensor_spec is not None
        return self._output_tensor_spec

    @property
    def mask_split_fn(self):
        """Returns the mask_split_fn for handling masked actions."""
        return self._mask_split_fn

    @property
    def variables(self):
        """Return the variables for all the network layers.

        If the network hasn't been built, builds it on random input (generated
        using self._input_tensor_spec) to build all the layers and their
        variables.

        Raises:
            ValueError:    If the network fails to build.
        """
        assert self.built
        return self.weights

    @property
    def trainable_variables(self):
        """Return the trainable variables for all the network layers.

        If the network hasn't been built, builds it on random input (generated
        using self._input_tensor_spec) to build all the layers and their
        variables.

        Raises:
            ValueError:    If the network fails to build.
        """
        assert self.built
        return self.trainable_weights

    @property
    def info_spec(self):
        return ()

    def copy(self, **kwargs):
        """Create a shallow copy of this network.

        **NOTE** Network layer weights are *never* copied.    This method
        recreates the `Network` instance with the same arguments it was
        initialized with (excepting any new kwargs).

        Args:
            **kwargs: Args to override when recreating this network.    Commonly
                overridden args include 'name'.

        Returns:
            A shallow copy of this network.
        """
        return type(self)(**dict(self._saved_kwargs, **kwargs))

    def __call__(self, inputs, *args, **kwargs):
        tf.nest.assert_same_structure(inputs, self.input_tensor_spec)
        # TODO: Debug.
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            outputs = self.call(inputs, *args, **kwargs)
            self._built = True
            return outputs


class DistributionNetwork(Network):
    """Base class for networks which generate Distributions as their output."""

    def __init__(self, input_tensor_spec, state_spec, output_spec, name):
        super(DistributionNetwork, self).__init__(
                input_tensor_spec=input_tensor_spec, state_spec=state_spec,
                name=name)
        self._output_spec = output_spec

    @property
    def output_spec(self):
        return self._output_spec
