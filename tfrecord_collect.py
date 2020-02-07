#!/usr/bin/env python

"""Collect TF-Record data for training the agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import os
import pprint
import random

import numpy as np
import tensorflow as tf
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common

from robovat.simulation.simulator import Simulator
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig

import policies
from utils import suite_env
from utils import tfrecord_replay_buffer


framework = tf.contrib.framework


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env',
        dest='env',
        type=str,
        help='The environment name.',
        required=True)

    parser.add_argument(
        '--env_config',
        dest='env_config',
        type=str,
        help='The configuration file for the environment.',
        default=None)

    parser.add_argument(
        '--policy',
        dest='policy',
        type=str,
        help='The policy name.',
        required=True)

    parser.add_argument(
        '--policy_config',
        dest='policy_config',
        type=str,
        help='The configuration file for the policy.',
        default=None)

    parser.add_argument(
        '--config_bindings',
        dest='config_bindings',
        type=str,
        help='The configuration bindings for the environment and the policy.',
        default=None)

    parser.add_argument(
        '--assets',
        dest='assets_dir',
        type=str,
        help='The assets directory.',
        default='./assets')

    parser.add_argument(
        '--rb_dir',
        dest='rb_dir',
        type=str,
        help='Directory of the replay buffer.',
        required=True)

    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        type=str,
        help='Directory of the checkpoint.',
        default=None)

    parser.add_argument(
        '--debug',
        dest='debug',
        type=int,
        help='Modifying the configuration.',
        default=0)

    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='None for random; any fixed integers for deterministic.',
        default=None)

    args = parser.parse_args()

    return args


def parse_config_files_and_bindings(args):
    """Parse the config files and the bindings.

    Args:
        args: The arguments.

    Returns:
        env_config: Environment configuration.
        policy_config: Policy configuration.
    """
    if args.env_config is None:
        env_config = dict()
    else:
        env_config = YamlConfig(args.env_config).as_easydict()

    if args.policy_config is None:
        policy_config = dict()
    else:
        policy_config = YamlConfig(args.policy_config).as_easydict()

    if args.config_bindings is not None:
        parsed_bindings = ast.literal_eval(args.config_bindings)
        logger.info('Config Bindings: %r', parsed_bindings)
        env_config.update(parsed_bindings)
        policy_config.update(parsed_bindings)

    logger.info('Environment Config:')
    pprint.pprint(env_config)
    logger.info('Policy Config:')
    pprint.pprint(policy_config)

    return env_config, policy_config


def collect(tf_env,
            tf_policy,
            output_dir,
            checkpoint=None,
            num_iterations=500000,
            episodes_per_file=500,
            summary_interval=1000):
    """A simple train and eval for SAC."""
    if not os.path.isdir(output_dir):
        logger.info('Making output directory %s...', output_dir)
        os.makedirs(output_dir)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

        # Make the replay buffer.
        replay_buffer = tfrecord_replay_buffer.TFRecordReplayBuffer(
                data_spec=tf_policy.trajectory_spec,
                experiment_id='exp',
                file_prefix=os.path.join(output_dir, 'data'),
                episodes_per_file=episodes_per_file)
        replay_observer = [replay_buffer.add_batch]

        collect_policy = tf_policy
        collect_op = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=replay_observer,
            num_steps=1).run()

        with tf.compat.v1.Session() as sess:
            # Initialize training.
            common.initialize_uninitialized_variables(sess)

            # Restore checkpoint.
            if checkpoint is not None:
                if os.path.isdir(checkpoint):
                    train_dir = os.path.join(checkpoint, 'train')
                    checkpoint_path = tf.train.latest_checkpoint(train_dir)
                else:
                    checkpoint_path = checkpoint

                restorer = tf.train.Saver(name='restorer')
                restorer.restore(sess, checkpoint_path)

            collect_call = sess.make_callable(collect_op)
            for _ in range(num_iterations):
                collect_call()


def main():
    args = parse_args()

    # Set the random seed.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # Configuration.
    env_config, policy_config = parse_config_files_and_bindings(args)

    # Create simulators.
    simulator = Simulator(assets_dir=args.assets_dir)

    # Create the environment.
    tf_env = tf_py_environment.TFPyEnvironment(
        suite_env.load(args.env, config=env_config, simulator=simulator))

    # Policy.
    policy_class = getattr(policies, args.policy)
    policy_config.update(tf_env.pyenv.envs[0].config)
    tf_policy = policy_class(time_step_spec=tf_env.time_step_spec(),
                             action_spec=tf_env.action_spec(),
                             config=policy_config)

    # Collect data.
    tf.compat.v1.enable_resource_variables()
    collect(tf_env=tf_env,
            tf_policy=tf_policy,
            output_dir=args.rb_dir,
            checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()
