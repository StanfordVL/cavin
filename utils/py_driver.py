"""A Driver that steps a python environment using a python policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
from tf_agents.drivers import driver
from tf_agents.trajectories import trajectory


def stack_nested_arrays(arrays):
    sample = arrays[0]
    if isinstance(sample, np.ndarray):
        return np.stack(arrays)
    elif isinstance(sample, list):
        return [np.stack([array[i] for array in arrays]) for i in len(sample)]
    elif isinstance(sample, (dict, OrderedDict)):
        return OrderedDict([
            (key, np.stack([array[key] for array in arrays])) for key in sample
        ])
    elif sample == ():
        return ()
    else:
        raise ValueError('Unrecognized: %r' % (type(sample)))


class PyStepDriver(driver.Driver):
    """A driver that runs a python policy in a python environment."""

    def __init__(self,
                 env,
                 policy,
                 observers,
                 max_steps=None,
                 max_episodes=None):
        """A driver that runs a python policy in a python environment.

        Args:
            env: A py_environment.Base environment.
            policy: A py_policy.Base policy.
            observers: A list of observers that are notified after every step
                in the environment. Each observer is a callable
                (trajectory.Trajectory).
            max_steps: Optional maximum number of steps for each run() call.
                Also see below. Default: 0.
            max_episodes: Optional maximum number of episodes for each run()
                call. At least one of max_steps or max_episodes must be
                provided. If both are set, run() terminates when at least one
                of the conditions is satisfied. Default: 0.

        Raises:
            ValueError: If both max_steps and max_episodes are None.
        """
        max_steps = max_steps or 0
        max_episodes = max_episodes or 0
        if max_steps < 1 and max_episodes < 1:
            raise ValueError('Either `max_steps` or `max_episodes` '
                             'should be greater than 0.')

        super(PyStepDriver, self).__init__(env, policy, observers)
        self._max_steps = max_steps or np.inf
        self._max_episodes = max_episodes or np.inf

    def run(self, time_step, policy_state=()):
        """Run policy in environment given initial time_step and policy_state.
        Args:
            time_step: The initial time_step.
            policy_state: The initial policy_state.
        Returns:
            A tuple (final time_step, final policy_state).
        """
        num_steps = 0
        num_episodes = 0
        while (num_steps < self._max_steps and
               num_episodes < self._max_episodes):
            if self.env.done:
                next_time_step = self.env.reset()
                next_policy_state = self.policy.get_initial_state()
            else:
                action_step = self.policy.action(time_step, policy_state)
                next_time_step = self.env.step(action_step.action)
                next_policy_state = action_step.state

                traj = trajectory.from_transition(
                    time_step, action_step, next_time_step)

                for observer in self.observers:
                    observer(traj)

                num_episodes += np.sum(traj.is_last())
                num_steps += np.sum(~traj.is_boundary())

            time_step = next_time_step
            policy_state = next_policy_state

        return time_step, policy_state


class PyEpisodeDriver(driver.Driver):
    """A driver that runs a python policy in a python environment."""

    def __init__(self,
                 env,
                 policy,
                 observers,
                 max_episodes):
        """A driver that runs a python policy in a python environment.

        Args:
            env: A py_environment.Base environment.
            policy: A py_policy.Base policy.
            observers: A list of observers that are notified after every step
                in the environment. Each observer is a callable
                (trajectory.Trajectory).
            max_episodes: Optional maximum number of episodes for each run()
                call. If both are set, run() terminates when at least one of
                the conditions is satisfied. Default: 0.

        Raises:
            ValueError: If both max_steps and max_episodes are None.
        """
        super(PyEpisodeDriver, self).__init__(env, policy, observers)

        max_episodes = max_episodes or 0
        self._max_episodes = max_episodes or np.inf

    def run(self, time_step, policy_state=()):
        """Run policy in environment given initial time_step and policy_state.
        Args:
            time_step: The initial time_step.
            policy_state: The initial policy_state.
        Returns:
            A tuple (final time_step, final policy_state).
        """
        for num_episodes in range(self._max_episodes):
            time_step = self.env.reset()
            policy_state = self.policy.get_initial_state()

            observation = []
            action = []
            policy_info = []
            reward = []

            while not self.env.done:
                action_step = self.policy.action(time_step, policy_state)
                if self.env.debug:
                    self.env.visualize(action_step.action, action_step.info)
                next_time_step = self.env.step(action_step.action)
                next_policy_state = action_step.state

                if len(self.observers) > 0:
                    observation.append(time_step.observation)
                    action.append(action_step.action)
                    policy_info.append(action_step.info)
                    reward.append(next_time_step.reward)

                time_step = next_time_step
                policy_state = next_policy_state

            if len(self.observers) > 0:
                # TODO: Find a better way than repeating the last action.
                observation.append(time_step.observation)
                action.append(action_step.action)
                policy_info.append(action_step.info)
                reward.append(next_time_step.reward)

                observation = stack_nested_arrays(observation)
                action = stack_nested_arrays(action)
                policy_info = stack_nested_arrays(policy_info)
                reward = stack_nested_arrays(reward)

                traj = trajectory.from_episode(
                    observation, action, policy_info, reward)

                for observer in self.observers:
                    observer(traj)

        return time_step, policy_state


class PyInitializationDriver(PyEpisodeDriver):
    """A driver that runs a python policy in a python environment."""

    def run(self, time_step, policy_state=()):
        """Run policy in environment given initial time_step and policy_state.
        Args:
            time_step: The initial time_step.
            policy_state: The initial policy_state.
        Returns:
            A tuple (final time_step, final policy_state).
        """
        for num_episodes in range(self._max_episodes):
            time_step = self.env.reset()
            policy_state = self.policy.get_initial_state()

            if len(self.observers) > 0:
                for observer in self.observers:
                    observer(time_step.observation)

        return time_step, policy_state
