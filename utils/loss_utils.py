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
    """L2 loss.

    Args:
        targets: The target tensor.
        outputs: The output tensor.
        weights: Optional Tensor to weight elements of the computed loss.
        reduction: Type of reduction to apply to loss.

    Returns:
        The loss tensor.
    """
    loss = 0.5 * tf.reduce_sum(tf.square(targets - outputs), axis=-1)
    return tf.losses.compute_weighted_loss(loss, weights, reduction=reduction)


def cosine_loss(targets,
                outputs,
                weights=1.0,
                reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Cosine loss.

    Args:
        targets: The target tensor.
        outputs: The output tensor.
        weights: Optional Tensor to weight elements of the computed loss.
        reduction: Type of reduction to apply to loss.

    Returns:
        The loss tensor.
    """
    targets_shape = targets.get_shape()
    outputs_shape = outputs.get_shape()
    assert len(targets_shape) == len(outputs_shape), (
        'Shapes are %r and %r.' % (targets_shape, outputs_shape))

    num_dims = len(targets_shape)
    for i in range(num_dims):
        assert targets_shape[i] == outputs_shape[i], (
            'Shapes are %r and %r.' % (targets_shape, outputs_shape))

    cosine_dist = (tf.reduce_sum(targets * outputs, axis=-1) /
                   tf.norm(targets, axis=-1) /
                   tf.norm(outputs, axis=-1))
    loss = 1.0 - cosine_dist
    return tf.losses.compute_weighted_loss(loss, weights, reduction=reduction)


def entropy(prob, weights=1.0):
    """Entropy of a probability.

    Args:
        prob: The input probability distribution.
        weights: Optional Tensor to weight elements of the computed loss.

    Returns:
        The entropy tensor.
    """
    assert prob.get_shape()[-1] > 0
    prob = tf.abs(prob) + EPS
    entropy = -tf.reduce_sum(prob * tf.log(prob), axis=-1)
    return tf.losses.compute_weighted_loss(entropy, weights)


def kl_divergence(p, q, weights=1.0):
    """Kullback–Leibler divergence between two distributions.

    Args:
        p: The first probability distribution.
        q: The second probability distribution.
        weights: Optional Tensor to weight elements of the computed loss.

    Returns:
        The result tensor.
    """
    assert p.get_shape()[-1] > 0
    assert q.get_shape()[-1] > 0
    p = tf.abs(p) + EPS
    q = tf.abs(q) + EPS
    loss = tf.reduce_sum(p * (tf.log(p) - tf.log(q)), axis=-1)
    return tf.losses.compute_weighted_loss(loss, weights)


def kl_divergence_gaussian(mean1, stddev1, mean2, stddev2, weights=1.0):
    """Kullback–Leibler divergence between two Gaussian distribution.

    Args:
        mean1: The mean of the first Gaussian distribution.
        stddev1: The standard deviation of the first Gaussian distribution.
        mean2: The mean of the second Gaussian distribution.
        stddev2: The standard deviation of the second Gaussian distribution.
        weights: Optional Tensor to weight elements of the computed loss.

    Returns:
        The result tensor.
    """
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
    """Helinger distance between two probability distributions.

    Args:
        p: The first probability distribution.
        q: The second probability distribution.
        weights: Optional Tensor to weight elements of the computed loss.

    Returns:
        The result tensor.
    """
    loss = 1.0 - tf.reduce_sum(tf.sqrt(p) * tf.sqrt(q), axis=-1)
    return tf.losses.compute_weighted_loss(loss, weights)


def log_normal(x, mean, stddev):
    """Compute the logorithmic normal distribution of a tensor.

    Args:
        x: The input tensor.
        mean: The mean of the Gaussian distribution.
        stddev: The standard deviation of the Gaussian distribution.

    Returns:
        The logorithmic normal distribution of the input tensor.
    """
    stddev = tf.abs(stddev)
    stddev = tf.add(stddev, EPS)
    return -0.5 * tf.reduce_sum(
      (tf.log(2 * np.pi) + tf.log(tf.square(stddev))
       + tf.square(x - mean) / tf.square(stddev)),
      axis=-1)
