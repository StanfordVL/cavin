from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from robovat.reward_fns import push_reward

from networks import network
from networks.pointnet import pointnet_encoder


nest = tf.contrib.framework.nest
slim = tf.contrib.slim


NORMALIZER_FN = None
NORMALIZER_PARAMS = None


def get_reward(states, next_states, task_name, layout_id, is_high_level=False):
    """Get the reward value.

    Args:
        states: The current states.
        next_states: The next states.
        task_name: Name of the task.
        layout_id: ID of the layout.
        is_high_level: If it is computed for high-level planning.

    Returns:
        rewards: The reward values.
        terminations: The termination flags.
    """
    assert layout_id is not None
    reward_fn = push_reward.get_reward_fn(
        task_name=task_name,
        layout_id=layout_id,
        is_planning=True,
        is_high_level=is_high_level)

    if isinstance(states, dict):
        states = states['position'][..., :2]
        next_states = next_states['position'][..., :2]

    rewards, terminations = tf.py_func(
        reward_fn,
        [states, next_states],
        [tf.float32, tf.bool])

    num_samples = int(states.shape[0])
    rewards = tf.reshape(rewards, [num_samples])
    terminations = tf.reshape(terminations, [num_samples])
    return rewards, terminations


def global_pool(input_tensor,
                axis,
                mask,
                mode='weighted_sum'):
    """Global pooling.

    Args:
        input_tensor: The input tensor.
        axis: The axis long which to pool.
        mask: Masks of valid entries.
        mode: One of 'weighted_sum', 'reduce_sum'.

    Returns:
        The output tensor.
    """
    if mode == 'weighted_sum':
        weight = slim.fully_connected(
            input_tensor,
            1,
            activation_fn=None,
            normalizer_fn=None,
            scope='weight')
        weight = tf.nn.softmax(weight, axis=axis)

        value = slim.fully_connected(
            input_tensor,
            int(input_tensor.shape[-1]),
            activation_fn=None,
            normalizer_fn=None,
            scope='value')
        value *= tf.expand_dims(mask, axis=-1)

        return tf.reduce_sum(weight * value, axis=axis, name='weighted_sum')

    elif mode == 'reduce_sum':
        value = tf.identity(input_tensor)
        value *= tf.expand_dims(mask, axis=-1)
        return tf.reduce_sum(value, axis=axis)

    else:
        raise ValueError('Unrecognized mode: %r', mode)


def encode_relation(positions,
                    body_masks,
                    dim_fc_state):
    """Encode the relation feature.

    Args:
        positions: Positions of the bodies.
        body_masks: Masks of valid bodies.
        dim_fc_state: Dimension of state encoding.

    Returns:
        A tensor of shape [batch_size, num_bodies, dim_fc_state].
    """
    with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=NORMALIZER_FN,
            normalizer_params=NORMALIZER_PARAMS):

        with tf.variable_scope('relation_masks'):
            body_masks = tf.identity(body_masks, 'body_masks')
            relation_masks = tf.subtract(
                tf.multiply(
                    tf.expand_dims(body_masks, -1),
                    tf.expand_dims(body_masks, -2)),
                tf.linalg.diag(body_masks))
            relation_masks = tf.expand_dims(relation_masks, axis=-1)

        with tf.variable_scope('relation_feats'):
            net = tf.subtract(
                tf.expand_dims(positions, axis=1),
                tf.expand_dims(positions, axis=2))
            net = slim.repeat(
                net,
                2,
                slim.fully_connected,
                dim_fc_state,
                scope='fc')
            relation_feats = net

    return tf.reduce_sum(relation_feats * relation_masks,
                         axis=1,
                         name='sum_relation_feats')


def encode_effect(states,
                  contexts,
                  use_relation,
                  use_point_cloud,
                  dim_fc_state,
                  dim_fc_context):
    """Encode the effect feature.

    Args:
        states: The state as a dict.
        contexts: The context data. Set to None if no contexts are used.
        use_relation: True if use relation encoding.
        use_point_cloud: True if point cloud data is used.
        dim_fc_state: Dimension of state encoding.
        dim_fc_context: Dimension of context encoding.

    Returns:
        A tensor of shape [batch_size, dim_fc_state].
    """
    positions = states['position']
    body_masks = states['body_mask']
    num_bodies = int(body_masks.shape[-1])

    with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=NORMALIZER_FN,
            normalizer_params=NORMALIZER_PARAMS):

        features = []

        with tf.variable_scope('encode_position'):
            position_feats = slim.fully_connected(
                positions, dim_fc_state, scope='fc')
            features.append(position_feats)

        if use_relation:
            with tf.variable_scope('encode_relation'):
                relation_feats = encode_relation(
                    positions,
                    body_masks,
                    dim_fc_state=dim_fc_state)
                features.append(relation_feats)

        if use_point_cloud:
            cloud_feats = states['cloud_feat']
            features.append(cloud_feats)

        if contexts is not None:
            with tf.variable_scope('encode_context'):
                context_feats = slim.fully_connected(
                    contexts, dim_fc_context, scope='fc')
                context_feats = tf.tile(tf.expand_dims(context_feats, 1),
                                        [1, num_bodies, 1])
                features.append(context_feats)

        net = tf.concat(features, axis=-1)
        net = slim.repeat(
            net,
            2,
            slim.fully_connected,
            dim_fc_state,
            scope='fc')
        effects = tf.identity(net, 'effects')

    return effects


def predict_state_with_effect(states,
                              effects,
                              dim_fc_state,
                              delta_position_range,
                              use_point_cloud):
    """Predict the state with the effect encoding.

    Args:
        states: The state as a dict.
        effects: The effect encoding.
        dim_fc_state: Dimension of state encoding.
        delta_position_range: Range of the position changes.
        use_point_cloud: True if point cloud data is used.

    Returns:
        The predicted state.
    """
    pred_states = collections.OrderedDict()
    pred_states['layout_id'] = tf.identity(states['layout_id'])
    pred_states['body_mask'] = tf.identity(states['body_mask'])

    with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=NORMALIZER_FN,
            normalizer_params=NORMALIZER_PARAMS):

        with tf.variable_scope('gate'):
            net = slim.fully_connected(
                effects, dim_fc_state, scope='fc')
            gate = slim.fully_connected(
                net,
                1,
                activation_fn=tf.sigmoid,
                normalizer_fn=None,
                scope='gate')

        with tf.variable_scope('pred_positions'):
            net = slim.fully_connected(
                effects, dim_fc_state, scope='fc')
            delta_positions = slim.fully_connected(
                net,
                2,
                activation_fn=None,
                normalizer_fn=None)
            delta_positions = tf.tanh(delta_positions) * delta_position_range
            delta_positions *= gate

            pred_positions = tf.add(
                states['position'], delta_positions)
            pred_states['position'] = pred_positions

        if use_point_cloud:
            with tf.variable_scope('pred_cloud_feats'):
                net = slim.fully_connected(
                    effects, dim_fc_state, scope='fc')
                delta_cloud_feats = slim.fully_connected(
                    net,
                    dim_fc_state,
                    activation_fn=None,
                    normalizer_fn=None)
                delta_cloud_feats *= gate

                pred_cloud_feats = tf.add(
                    states['cloud_feat'], delta_cloud_feats)
                pred_cloud_feats /= (1e-16 + tf.norm(
                    pred_cloud_feats, axis=-1, keepdims=True))

                pred_states['cloud_feat'] = pred_cloud_feats

    return pred_states


def generate_action(inputs, dim_fc_action, num_goal_steps):
    """Generate the action for multiple steps.

    Args:
        inputs: The input tensor.
        dim_fc_action: Dimension of action encoding.
        num_goal_steps: Number of goal steps.

    Returns:
        The generated action.
    """
    with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=NORMALIZER_FN,
            normalizer_params=NORMALIZER_PARAMS):

        with tf.variable_scope('starts'):
            net = inputs
            net = slim.fully_connected(
                net, dim_fc_action, scope='fc')
            starts = slim.fully_connected(
                net,
                num_goal_steps * 2,
                activation_fn=None,
                normalizer_fn=None,
                scope='starts')
            starts = tf.identity(tf.tanh(starts / 5.0) * 5.0,
                                 'softly_clipped_starts')
            starts = tf.reshape(
                starts,
                [-1, num_goal_steps, 2])

        with tf.variable_scope('motions'):
            net = inputs
            net = slim.fully_connected(
                net, dim_fc_action, scope='fc')
            motions = slim.fully_connected(
                net,
                num_goal_steps * 2,
                activation_fn=tf.tanh,
                normalizer_fn=None,
                scope='motions')
            motions = tf.reshape(
                motions,
                [-1, num_goal_steps, 2])

    actions = tf.concat([starts, motions], -1, name='actions')
    return actions


def generate_action_single_step(inputs, dim_fc_action):
    """Generate the action for single steps.

    Args:
        inputs: The input tensor.
        dim_fc_action: Dimension of action encoding.

    Returns:
        The generated action.
    """
    with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=tf.nn.relu,
            normalizer_fn=NORMALIZER_FN,
            normalizer_params=NORMALIZER_PARAMS):

        with tf.variable_scope('starts'):
            net = inputs
            net = slim.fully_connected(
                net, dim_fc_action, scope='fc')
            starts = slim.fully_connected(
                net,
                2,
                activation_fn=None,
                normalizer_fn=None,
                scope='starts')
            starts = tf.identity(tf.tanh(starts / 5.0) * 5.0,
                                 'softly_clipped_starts')

        with tf.variable_scope('motions'):
            net = inputs
            net = slim.fully_connected(
                net, dim_fc_action, scope='fc')
            motions = slim.fully_connected(
                net,
                2,
                activation_fn=tf.tanh,
                normalizer_fn=None,
                scope='motions')

    actions = tf.concat([starts, motions], -1, name='actions')
    return actions


class StateEncoder(network.Network):
    """State encoder."""

    def __init__(self,
                 input_tensor_spec,
                 config,
                 name='StateEncoder'):
        """Initialize."""
        super(StateEncoder, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        self._dim_fc_state = config.DIM_FC_STATE

    def call(self, inputs, step_type=None, network_states=()):
        del step_type    # unused.

        observations = inputs
        positions = observations['position'][..., :2]
        point_clouds = observations['point_cloud']

        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_cloud'):
                offsets = tf.concat(
                    [positions, tf.zeros_like(positions[..., 0:1])],
                    axis=-1)
                offsets = tf.expand_dims(offsets, axis=-2)
                centered_point_clouds = tf.subtract(
                    point_clouds, offsets, name='centered_point_clouds')
                cloud_feats, _ = pointnet_encoder(
                    centered_point_clouds,
                    conv_layers=[16, 32, 64],
                    fc_layers=[],
                    dim_output=self._dim_fc_state,
                    normalizer_fn=NORMALIZER_FN,
                    normalizer_params=NORMALIZER_PARAMS,
                    scope='pointnet')
                cloud_feats /= (1e-16 + tf.norm(
                    cloud_feats, axis=-1, keepdims=True))

        states = collections.OrderedDict()
        states['layout_id'] = observations['layout_id']
        states['body_mask'] = observations['body_mask']
        states['position'] = positions
        states['cloud_feat'] = cloud_feats
        return states


class EffectEncoder(network.Network):
    """Effect encoder."""

    def __init__(self,
                 input_tensor_spec,
                 config,
                 name='EffectEncoder'):
        """Initialize."""
        super(EffectEncoder, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._use_point_cloud = config.USE_POINT_CLOUD
        self._use_relation = config.USE_RELATION
        self._dim_fc_state = config.DIM_FC_STATE
        self._dim_fc_c = config.DIM_FC_C

    def call(self, inputs, step_type=None, network_states=()):
        del step_type    # unused.

        states, cs = inputs
        return encode_effect(
            states=states,
            contexts=cs,
            use_relation=self._use_relation,
            use_point_cloud=self._use_point_cloud,
            dim_fc_state=self._dim_fc_state,
            dim_fc_context=self._dim_fc_c)


class EffectEncoderWithoutC(network.Network):
    """Effect encoder without using c."""

    def __init__(self,
                 input_tensor_spec,
                 config,
                 name='EffectEncoder'):
        """Initialize."""
        super(EffectEncoderWithoutC, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._use_point_cloud = config.USE_POINT_CLOUD
        self._use_relation = config.USE_RELATION
        self._dim_fc_state = config.DIM_FC_STATE

    def call(self, inputs, step_type=None, network_states=()):
        del step_type    # unused.

        states = inputs
        return encode_effect(
            states=states,
            contexts=None,
            use_relation=self._use_relation,
            use_point_cloud=self._use_point_cloud,
            dim_fc_state=self._dim_fc_state,
            dim_fc_context=None)


class Dynamics(network.Network):
    """Dynamics model."""

    def __init__(self,
                 input_tensor_spec,
                 config,
                 name='Dynamics'):
        """Initialize."""
        super(Dynamics, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._use_point_cloud = config.USE_POINT_CLOUD
        self._use_relation = config.USE_RELATION
        self._dim_fc_state = config.DIM_FC_STATE
        self._dim_fc_action = config.DIM_FC_ACTION
        self._delta_position_range = config.DELTA_STATE_POSITION_RANGE

    def call(self, inputs, step_type=None, network_states=()):
        del step_type    # unused.

        states, actions = inputs

        pred_states = collections.OrderedDict()
        pred_states['layout_id'] = tf.identity(states['layout_id'])
        pred_states['body_mask'] = tf.identity(states['body_mask'])

        with tf.variable_scope('clip_action'):
            actions = tf.clip_by_value(actions, -1., 1., 'clipped_actions')

        with tf.variable_scope('encode_effect'):
            effects = encode_effect(
                states=states,
                contexts=actions,
                use_relation=self._use_relation,
                use_point_cloud=self._use_point_cloud,
                dim_fc_state=self._dim_fc_state,
                dim_fc_context=self._dim_fc_action)

        return predict_state_with_effect(
            states=states,
            effects=effects,
            dim_fc_state=self._dim_fc_state,
            delta_position_range=self._delta_position_range,
            use_point_cloud=self._use_point_cloud)


class MetaDynamics(network.Network):
    """Meta dyanmics model."""

    def __init__(self,
                 input_tensor_spec,
                 config,
                 name='MetaDynamics'):
        """Initialize."""
        super(MetaDynamics, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._use_point_cloud = config.USE_POINT_CLOUD
        self._use_relation = config.USE_RELATION
        self._dim_fc_state = config.DIM_FC_STATE
        self._dim_fc_action = config.DIM_FC_ACTION
        self._delta_position_range = config.DELTA_GOAL_POSITION_RANGE

    def call(self, inputs, step_type=None, network_states=()):
        del step_type    # unused.

        states, effects = inputs

        return predict_state_with_effect(
            states=states,
            effects=effects,
            dim_fc_state=self._dim_fc_state,
            delta_position_range=self._delta_position_range,
            use_point_cloud=self._use_point_cloud)


class ActionGenerator(network.Network):
    """Action generator."""

    def __init__(self,
                 input_tensor_spec,
                 config,
                 use_single_step=False,
                 name='ActionGenerator'):
        """Initialize."""
        super(ActionGenerator, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._dim_z = config.DIM_Z
        self._dim_fc_z = config.DIM_FC_Z
        self._dim_fc_action = config.DIM_FC_ACTION
        self._num_goal_steps = config.NUM_GOAL_STEPS

        self._use_single_step = use_single_step

    def call(self, inputs, step_type=None, network_states=()):
        del step_type    # unused.

        states, effects, zs = inputs

        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('global_pool'):
                effects = global_pool(
                    effects,
                    axis=1,
                    mask=states['body_mask'],
                    mode='weighted_sum')

            with tf.variable_scope('transform_z'):
                net = effects
                net = slim.fully_connected(
                    net, self._dim_fc_z, scope='fc')
                gaussian_params = slim.fully_connected(
                    net,
                    2 * self._dim_z,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                means = tf.identity(
                    gaussian_params[..., :self._dim_z], name='means')
                stddevs = tf.add(
                    tf.nn.softplus(gaussian_params[..., self._dim_z:]),
                    1e-6,
                    name='stddevs')
                transformed_zs = tf.identity(means + stddevs * zs,
                                             'transformed_zs')

        if self._use_single_step:
            return generate_action_single_step(
                transformed_zs,
                dim_fc_action=self._dim_fc_action)
        else:
            return generate_action(
                transformed_zs,
                dim_fc_action=self._dim_fc_action,
                num_goal_steps=self._num_goal_steps)


class ActionGeneratorWithoutZ(network.Network):
    """Action generator without Z."""

    def __init__(self,
                 input_tensor_spec,
                 config,
                 use_single_step=False,
                 name='ActionGenerator'):
        """Initialize."""
        super(ActionGeneratorWithoutZ, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._dim_fc_action = config.DIM_FC_ACTION
        self._num_goal_steps = config.NUM_GOAL_STEPS

        self._use_single_step = use_single_step

    def call(self, inputs, step_type=None, network_states=()):
        del step_type    # unused.

        states, effects = inputs

        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('global_pool'):
                effects = global_pool(
                    effects,
                    axis=1,
                    mask=states['body_mask'],
                    mode='weighted_sum')

        if self._use_single_step:
            return generate_action_single_step(
                effects,
                dim_fc_action=self._dim_fc_action)
        else:
            return generate_action(
                effects,
                dim_fc_action=self._dim_fc_action,
                num_goal_steps=self._num_goal_steps)


class CInferenceNetwork(network.Network):
    """Inference network of c."""

    def __init__(self,
                 input_tensor_spec,
                 config,
                 is_sectar=False,
                 name='CInferenceNetwork'):
        """Initialize."""
        super(CInferenceNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        if is_sectar:
            self._dim_c = config.DIM_C + config.DIM_Z
        else:
            self._dim_c = config.DIM_C

        self._dim_fc_state = config.DIM_FC_STATE

    def call(self, inputs, step_type=None, network_states=()):
        del step_type    # unused.

        states, goals = inputs

        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_dynamics'):
                positions = states['position']
                next_positions = goals['position']
                delta_positions = tf.subtract(
                    next_positions,
                    positions,
                    'delta_positions')
                net = tf.concat([positions, delta_positions], axis=-1)
                net = slim.repeat(
                    net,
                    2,
                    slim.fully_connected,
                    self._dim_fc_state,
                    scope='fc')
                dynamics_feats = tf.identity(net, 'dynamics_feats')

                with tf.variable_scope('global_pool'):
                    dynamics_feats = global_pool(
                        dynamics_feats,
                        axis=1,
                        mask=states['body_mask'],
                        mode='reduce_sum')

            with tf.variable_scope('inference'):
                net = dynamics_feats
                net = slim.fully_connected(
                    net, self._dim_fc_state, scope='fc')
                gaussian_params = slim.fully_connected(
                    net,
                    2 * self._dim_c,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                c_means = tf.identity(
                    gaussian_params[..., :self._dim_c], name='c_means')
                c_stddevs = tf.add(
                    tf.nn.softplus(gaussian_params[..., self._dim_c:]),
                    1e-6,
                    name='c_stddevs')

        return c_means, c_stddevs


class ZInferenceNetwork(network.Network):
    """Inference network of z."""

    def __init__(self,
                 input_tensor_spec,
                 config,
                 use_single_step=False,
                 name='ZInferenceNetwork'):
        """Initialize."""
        super(ZInferenceNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._dim_z = config.DIM_Z
        self._dim_fc_z = config.DIM_FC_Z
        self._dim_fc_action = config.DIM_FC_ACTION
        self._num_goal_steps = config.NUM_GOAL_STEPS

        self._use_single_step = use_single_step

    def call(self, inputs, step_type=None, network_states=()):
        del step_type    # unused.

        states, actions, effects = inputs

        if not self._use_single_step:
            assert actions.shape[-2] == self._num_goal_steps

        with slim.arg_scope(
                [slim.fully_connected],
                activation_fn=tf.nn.relu,
                normalizer_fn=NORMALIZER_FN,
                normalizer_params=NORMALIZER_PARAMS):

            with tf.variable_scope('encode_action'):
                action_feats = slim.fully_connected(
                    actions, self._dim_fc_action, scope='fc')
                _shape = action_feats.shape.as_list()
                if not self._use_single_step:
                    action_feats = tf.reshape(
                        action_feats,
                        [-1, _shape[-2] * _shape[-1]])

            with tf.variable_scope('global_pool'):
                effects = global_pool(
                    effects,
                    axis=1,
                    mask=states['body_mask'],
                    mode='weighted_sum')

            with tf.variable_scope('inference'):
                net = tf.concat([effects, action_feats], axis=-1)
                net = slim.fully_connected(
                    net, self._dim_fc_z, scope='fc')
                gaussian_params = slim.fully_connected(
                    net,
                    2 * self._dim_z,
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='gaussian_params')
                z_means = tf.identity(
                    gaussian_params[..., :self._dim_z], name='z_means')
                z_stddevs = tf.add(
                    tf.nn.softplus(gaussian_params[..., self._dim_z:]),
                    1e-6,
                    name='z_stddevs')

        return z_means, z_stddevs
