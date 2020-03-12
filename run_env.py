#!/usr/bin/env python

"""Run an environment with the chosen policy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import os
import random
import pkg_resources  # NOQA
import pprint

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common as common_utils

from robovat.simulation.simulator import Simulator
from robovat.utils.logging import logger
from robovat.utils.yaml_config import YamlConfig

from utils import suite_env
from utils import py_driver


tf.compat.v1.disable_eager_execution()


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
        '--use_simulator',
        dest='use_simulator',
        type=int,
        help='Run experiments in the simulation is it is True.',
        required=False,
        default=1)

    parser.add_argument(
        '--assets',
        dest='assets_dir',
        type=str,
        help='The assets directory.',
        default='./assets')

    parser.add_argument(
        '--worker_id',
        dest='worker_id',
        type=int,
        help='The worker ID for running multiple simulations in parallel.',
        default=0)

    parser.add_argument(
        '--debug',
        dest='debug',
        type=int,
        help='Use the debugging mode if it is True.',
        default=0)

    parser.add_argument(
        '--num_episodes',
        dest='num_episodes',
        type=int,
        help='Maximum number of episodes.',
        default=1000000)

    parser.add_argument(
        '--episodic',
        dest='episodic',
        type=int,
        help='If True, use episode driver, otherwise use the step driver.',
        default=1)

    parser.add_argument(
        '--use_random_policy',
        dest='use_random_policy',
        type=int,
        help='If True, use the random policy.',
        default=None)

    parser.add_argument(
        '--policy',
        dest='policy',
        type=str,
        help='The policy name.',
        default=None)

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
        '--checkpoint',
        dest='checkpoint',
        type=str,
        help='Directory of the checkpoint.',
        default=None)

    parser.add_argument(
        '--problem',
        dest='problem',
        type=str,
        help='Name of the problem.',
        default=None)

    parser.add_argument(
        '--seed',
        dest='seed',
        type=int,
        help='None for random; any fixed integers for deterministic.',
        default=None)

    parser.add_argument(
        '--output',
        dest='output_dir',
        type=str,
        help='Directory of the output data.',
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


def main():
    # pybullet_version = pkg_resources.get_distribution("pybullet").version
    # assert pybullet_version == '1.8.0', (
        # 'Please use pybullet version \'1.8.0\'. The current version %r '
        # 'could lead to different simulation results.')

    args = parse_args()

    # Set the random seed.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # Configuration.
    env_config, policy_config = parse_config_files_and_bindings(args)

    # Simulator.
    if args.use_simulator:
        simulator = Simulator(worker_id=args.worker_id,
                              use_visualizer=bool(args.debug),
                              assets_dir=args.assets_dir)
    else:
        simulator = None

    # Environment.
    logger.info('Building the environment %s...', args.env)

    py_env = suite_env.load(args.env,
                            simulator=simulator,
                            config=env_config,
                            debug=args.debug,
                            max_episode_steps=None)

    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    # Policy.
    if args.use_random_policy:
        tf_policy = random_tf_policy.RandomTFPolicy(
            time_step_spec=tf_env.time_step_spec(),
            action_spec=tf_env.action_spec(),
        )
    else:
        assert args.policy is not None
        import policies
        policy_class = getattr(policies, args.policy)
        tf_policy = policy_class(time_step_spec=tf_env.time_step_spec(),
                                 action_spec=tf_env.action_spec(),
                                 config=policy_config)

    py_policy = py_tf_policy.PyTFPolicy(tf_policy)

    # This is required by tf-agents-nightly.
    tf.compat.v1.enable_resource_variables()

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        observers = []

        # Generate episodes.
        if args.episodic:
            driver = py_driver.PyEpisodeDriver(
                py_env,
                py_policy,
                observers=observers,
                max_episodes=args.num_episodes)
        else:
            driver = py_driver.PyStepDriver(
                py_env,
                py_policy,
                observers=observers,
                max_steps=None,
                max_episodes=args.num_episodes)

        # Initialize.
        if len(tf_policy.variables()) > 0:
            common_utils.initialize_uninitialized_variables(sess)

        if args.checkpoint is not None:
            if os.path.isdir(args.checkpoint):
                train_dir = os.path.join(args.checkpoint, 'train')
                checkpoint_path = tf.compat.v1.train.latest_checkpoint(
                    train_dir)
            else:
                checkpoint_path = args.checkpoint

            restorer = tf.compat.v1.train.Saver(name='restorer')
            logger.info('Restoring parameters from %s ...', checkpoint_path)
            restorer.restore(sess, checkpoint_path)

        # Run.
        time_step = py_env.reset()
        # policy_state = py_policy.get_initial_state(py_env.batch_size)
        policy_state = ()
        driver.run(time_step, policy_state)


if __name__ == '__main__':
    main()
