"""Tensorflow policies.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.policies import tf_policy


class TFPolicy(tf_policy.Base):
    """CEM policy for multi-stage tasks."""

    def __init__(self,
                 time_step_spec,
                 action_spec,
                 policy_state_spec=(),
                 info_spec=(),
                 config=None):
        """Initialize"""
        super(TFPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            info_spec=info_spec)
        self.config = config
