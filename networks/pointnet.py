"""PointNet.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


slim = tf.contrib.slim


NORMALIZER_FN = slim.batch_norm
NORMALIZER_PARAMS = {
    'decay': 0.997,
    'epsilon': 1e-5,
    'variables_collections': {
                    'beta': None,
                    'gamma': None,
                    'moving_mean': ['moving_vars'],
                    'moving_variance': ['moving_vars'],
                    },
    'center': True,
    'scale': True,
}


def pointnet_encoder(
        point_cloud,
        conv_layers=[16, 32, 64],
        fc_layers=[128],
        dim_output=128,
        dropout_rate=0.5,
        normalizer_fn=NORMALIZER_FN,
        normalizer_params=NORMALIZER_PARAMS,
        is_training=False,
        scope=None):
    """PointNet encoder for feature extraction.

    Args:
        point_cloud: The input point cloud.
        conv_layers: List of number of channels for convolutional layers.
        fc_layers: List of dimensions for fully-connected layers.
        dim_output: Output dimension.
        dropout_rate: Dropout rate of the fully-connected layers.
        normalizer_fn: Normalizer function.
        normalizer_params: Normalizer parameters.
        is_training: True if it is the training mode.
        scope: Scope of the module.

    Returns:
        net: A tensor of shape [batch_size, dim_output].
        end_points: List of intermediate layers.
    """
    end_points = {}

    with tf.variable_scope(scope, default_name='pointnet',
                           reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.dropout, slim.batch_norm],
                                is_training=is_training):
                with slim.arg_scope(
                        [slim.conv2d, slim.fully_connected],
                        weights_initializer=(
                            tf.contrib.layers.xavier_initializer()),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params):

                    net = point_cloud

                    net = slim.stack(net, slim.fully_connected, conv_layers,
                                     scope='conv')
                    end_points['final_conv'] = net

                    net = tf.reduce_max(net, axis=-2, name='max_pool')
                    end_points['max_pool'] = net

                    if fc_layers:
                        for i, num_filters in enumerate(fc_layers):
                            layer_name = 'fc%d' % (i + 1)
                            net = slim.fully_connected(
                                net, 128, scope=layer_name)
                            end_points[layer_name] = net

                            if dropout_rate is not None:
                                layer_name = 'dropout%d' % (i + 1)
                                keep_prob = 1.0 - dropout_rate
                                net = slim.dropout(
                                    net, keep_prob=keep_prob, scope=layer_name)
                                end_points[layer_name] = net

                    net = slim.fully_connected(
                        net,
                        dim_output,
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='output')
                    end_points['output'] = net

    return net, end_points
