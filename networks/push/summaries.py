"""Summaries for pushing tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def state_summary(scope, states, step):
    """Summary of the state.

    Args:
        scope: Scope of the summary.
        states: The current states.
        step: The training step.
    """
    with tf.name_scope(scope):
        position = states['position']
        tf.compat.v2.summary.histogram('x', position[..., 0], step=step)
        tf.compat.v2.summary.histogram('y', position[..., 1], step=step)


def delta_state_summary(scope, states, next_states, step):
    """Summary of the state changes.

    Args:
        scope: Scope of the summary.
        states: The current states.
        next_states: The next states.
        step: The training step.
    """
    with tf.name_scope(scope):
        delta = next_states['position'] - states['position']
        tf.compat.v2.summary.histogram('dx', delta[..., 0], step=step)
        tf.compat.v2.summary.histogram('dy', delta[..., 1], step=step)
        tf.compat.v2.summary.histogram('l2_dist', tf.norm(delta, axis=-1),
                                       step=step)


def action_summary(scope, actions, step):
    """Summary of the action.

    Args:
        scope: Scope of the summary.
        actions: The actions.
        step: The training step.
    """
    assert int(actions.shape[-1]) == 4
    with tf.name_scope(scope):
        tf.compat.v2.summary.histogram('start', actions[..., :2], step=step)
        tf.compat.v2.summary.histogram('motion', actions[..., 2:], step=step)


def delta_action_summary(scope, actions_1, actions_2, step):
    """Summary of the delta between two actions.

    Args:
        scope: Scope of the summary.
        actions_1: The first actions.
        actions_2: The second actions.
        step: The training step.
    """
    assert int(actions_1.shape[-1]) == 4
    assert int(actions_2.shape[-1]) == 4
    with tf.name_scope(scope):
        delta = actions_2 - actions_1
        tf.compat.v2.summary.histogram('d_start', delta[..., :2], step=step)
        tf.compat.v2.summary.histogram('d_motion', delta[..., 2:], step=step)
