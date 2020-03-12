"""Push policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.trajectories import policy_step

from networks.push import samplers
from policies import tf_policy


class HeuristicPushPolicy(tf_policy.TFPolicy):
    """Random policy using modes"""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 policy_state_spec=(),
                 info_spec=(),
                 config=None):
        self._sampler = samplers.HeuristicPushSampler(
            input_tensor_spec=time_step_spec.observation,
            output_tensor_spec=action_spec,
            config=config)

        super(HeuristicPushPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=policy_state_spec,
            info_spec=info_spec,
            config=config)

    def _variables(self):
        return []

    def _action(self, time_step, policy_state, seed):
        action = self._sampler(time_step.observation, 1, seed)
        return policy_step.PolicyStep(action, policy_state)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError(
            'RandomTFPolicy does not support distributions yet.')
