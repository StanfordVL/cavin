"""Loss utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim
Reduction = tf.losses.Reduction


EPS = 1e-20


def l2_loss(targets,
            outputs,
            weights=1.0,
            reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    loss = 0.5 * tf.reduce_sum(tf.square(targets - outputs), axis=-1)
    return tf.losses.compute_weighted_loss(loss, weights, reduction=reduction)


def angle_loss(targets, outputs, weights=1.0,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    targets_shape = targets.get_shape()
    outputs_shape = outputs.get_shape()
    assert len(targets_shape) == len(outputs_shape), (
        'Shapes are %r and %r.' % (targets_shape, outputs_shape))

    num_dims = len(targets_shape)
    for i in range(num_dims):
        assert targets_shape[i] == outputs_shape[i], (
            'Shapes are %r and %r.' % (targets_shape, outputs_shape))

    cos_dist = (tf.reduce_sum(targets * outputs, axis=-1) /
                tf.norm(targets, axis=-1) /
                tf.norm(outputs, axis=-1))
    loss = 1.0 - cos_dist
    return tf.losses.compute_weighted_loss(loss, weights, reduction=reduction)


def entropy(prob, weights=1.0):
    assert prob.get_shape()[-1] > 0
    prob = tf.abs(prob) + EPS
    entropy = -tf.reduce_sum(prob * tf.log(prob), axis=-1)
    return tf.losses.compute_weighted_loss(entropy, weights)


def kl_divergence(p, q, weights=1.0):
    assert p.get_shape()[-1] > 0
    assert q.get_shape()[-1] > 0
    p = tf.abs(p) + EPS
    q = tf.abs(q) + EPS
    loss = tf.reduce_sum(p * (tf.log(p) - tf.log(q)), axis=-1)
    return tf.losses.compute_weighted_loss(loss, weights)


def kl_divergence_gaussian(mean1, stddev1, mean2, stddev2, weights=1.0):
    loss = (
        - 0.5 * tf.ones_like(mean1)
        + tf.log(stddev2 + EPS)
        - tf.log(stddev1 + EPS)
        + 0.5 * tf.square(stddev1) / (tf.square(stddev2) + EPS)
        + 0.5 * tf.square(mean2 - mean1) / (tf.square(stddev2) + EPS)
    )
    loss = tf.reduce_sum(loss, axis=-1)
    return tf.losses.compute_weighted_loss(loss, weights)


def hellinger_distance(p, q, weights=1.0):
    loss = 1.0 - tf.reduce_sum(tf.sqrt(p) * tf.sqrt(q), axis=-1)
    return tf.losses.compute_weighted_loss(loss, weights)


def log_normal(x, mean, stddev):
    stddev = tf.abs(stddev)
    stddev = tf.add(stddev, EPS)

    return -0.5 * tf.reduce_sum(
      (tf.log(2 * np.pi) + tf.log(tf.square(stddev))
       + tf.square(x - mean) / tf.square(stddev)),
      axis=-1)
