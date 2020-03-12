"""Losses for pushing tasks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from utils import loss_utils


MIN_STATE_DIST = 0.05
MAX_STATE_DIST = 0.2

MIN_GOAL_DIST = 0.1
MAX_GOAL_DIST = 0.5


def get_dynamics_weights(time_steps, actions, next_time_steps):
    """Get the weights for training the dynamics model.

    Args:
        time_steps: The current time step.
        actions: The actions.
        next_time_steps: The next time step.

    Returns:
        A tensor of [batch_size, num_objects].
    """
    position = time_steps.observation['position']
    next_position = next_time_steps.observation['position']
    dists = tf.norm(next_position - position, axis=-1)

    max_valid = tf.less(dists, MAX_STATE_DIST)
    valid = tf.reduce_all(max_valid, axis=-1)

    return tf.cast(valid, tf.float32, 'dynamics_weights')


def get_c_weights(time_steps, actions, next_time_steps):
    """Get the weights for c.

    Args:
        time_steps: The current time step.
        actions: The actions.
        next_time_steps: The next time step.

    Returns:
        A tensor of [batch_size].
    """
    position = time_steps.observation['position'][:, 0]
    next_position = next_time_steps.observation['position'][:, -1]
    dists = tf.norm(next_position - position, axis=-1)

    min_valid = tf.greater(dists, MIN_GOAL_DIST)
    max_valid = tf.less(dists, MAX_GOAL_DIST)
    valid = tf.logical_and(
        tf.reduce_any(min_valid, axis=-1),
        tf.reduce_all(max_valid, axis=-1))

    is_safe = next_time_steps.observation['is_safe']
    is_safe = tf.cast(is_safe, tf.bool)
    is_safe = tf.reduce_all(is_safe, axis=-1)
    valid = tf.logical_and(valid, tf.cast(is_safe, tf.bool))

    return tf.cast(valid, tf.float32, 'c_weights')


def get_z_weights(time_steps, actions, next_time_steps):
    """Get the weights for z.

    Args:
        time_steps: The current time step.
        actions: The actions.
        next_time_steps: The next time step.

    Returns:
        A tensor of [batch_size].
    """
    position = time_steps.observation['position']
    next_position = next_time_steps.observation['position']
    dists = tf.norm(next_position - position, axis=-1)

    min_valid = tf.greater(dists, MIN_STATE_DIST)
    max_valid = tf.less(dists, MAX_STATE_DIST)
    valid = tf.logical_and(
        tf.reduce_any(min_valid, axis=-1),
        tf.reduce_all(max_valid, axis=-1))
    valid = tf.reduce_all(max_valid, axis=-1)

    is_safe = next_time_steps.observation['is_safe']
    valid = tf.logical_and(valid, tf.cast(is_safe, tf.bool))

    return tf.cast(valid, tf.float32, 'z_weights')


def _state_loss(target_data,
                output_data,
                body_mask,
                weights=1.0):
    """Compute the loss of a state property.

    Args:
        target_data: The target value of the state.
        output_data: The output value of the state.
        body_mask: Masks of valid bodies.
        weights: Weights of the loss.

    Returns:
        The weighted Euclidean distance.
    """
    num_bodies = int(target_data.shape[-2])
    target_data *= tf.expand_dims(body_mask, axis=-1)
    output_data *= tf.expand_dims(body_mask, axis=-1)

    loss = 0.0
    for i in range(num_bodies):
        diff = target_data[..., i, :] - output_data[..., i, :]
        loss += 0.5 * tf.reduce_sum(tf.square(diff), axis=-1)

    return tf.compat.v1.losses.compute_weighted_loss(loss, weights)


def state_loss(targets, outputs, weights=1.0):
    """Compute the state loss.

    Args:
        targets: The target state.
        outputs: The output state.
        weights: Weights of the loss.

    Returns:
        The state loss.
    """
    return _state_loss(target_data=targets['position'],
                       output_data=outputs['position'],
                       body_mask=targets['body_mask'],
                       weights=weights)


def state_encoding_loss(targets, outputs, weights=1.0):
    """Compute the state encoding loss.

    Args:
        targets: The target state.
        outputs: The output state.
        weights: Weights of the loss.

    Returns:
        The state encoding loss.
    """
    return _state_loss(target_data=targets['cloud_feat'],
                       output_data=outputs['cloud_feat'],
                       body_mask=targets['body_mask'],
                       weights=weights)


def action_loss(targets, outputs, weights=1.0):
    """Compute the action loss.

    Args:
        targets: The target action.
        outputs: The output action.
        weights: Weights of the loss.

    Returns:
        The action loss.
    """
    assert len(targets.shape) == len(outputs.shape)
    # Weight starting position by 10.
    return 10.0 * loss_utils.l2_loss(
        targets=targets[..., :2],
        outputs=outputs[..., :2],
        weights=weights) + loss_utils.l2_loss(
            targets=targets[..., 2:],
            outputs=outputs[..., 2:],
            weights=weights)
