#!/usr/bin/env python

"""Use the TF-Record data to train and eval the agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import os
import pprint
import random
import time
from absl import logging

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.utils import common

from robovat.simulation.simulator import Simulator
from robovat.utils.yaml_config import YamlConfig

from agents import mpc_agent
from agents import vae_mpc_agent
from agents import sectar_agent
from agents import cavin_agent
from utils import suite_env
from utils import tfrecord_replay_buffer


framework = tf.contrib.framework


AGENT_CTORS = {
    'mpc': mpc_agent.MpcAgent,
    'vae': vae_mpc_agent.VaeMpcAgent,
    'sectar': sectar_agent.SectarAgent,
    'cavin': cavin_agent.CavinAgent,
}


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
        '--agent',
        dest='agent',
        type=str,
        help='The agent name.',
        default=None)

    parser.add_argument(
        '--env_config',
        dest='env_config',
        type=str,
        help='The configuration file for the environment.',
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
        '--assets',
        dest='assets_dir',
        type=str,
        help='The assets directory.',
        default='./assets')

    parser.add_argument(
        '--working_dir',
        dest='working_dir',
        type=str,
        help='Working directory',
        default=None)

    parser.add_argument(
        '--rb_dir',
        dest='rb_dir',
        type=str,
        help='Directory of the replay buffer.',
        default=None)

    parser.add_argument(
        '--lr',
        dest='learning_rate',
        type=float,
        help='The learning rate',
        default=1e-4)

    parser.add_argument(
        '--debug',
        dest='debug',
        type=int,
        help='Modifying the configuration.',
        default=0)

    parser.add_argument(
        '--output',
        dest='output_dir',
        type=str,
        help='Directory of the output data.',
        default=None)

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
        env_config = None
    else:
        env_config = YamlConfig(args.env_config).as_easydict()

    if args.policy_config is None:
        policy_config = None
    else:
        policy_config = YamlConfig(args.policy_config).as_easydict()

    if args.config_bindings is not None:
        parsed_bindings = ast.literal_eval(args.config_bindings)
        logging.info('Config Bindings: %r', parsed_bindings)
        env_config.update(parsed_bindings)
        policy_config.update(parsed_bindings)

    logging.info('Environment Config:')
    pprint.pprint(env_config)
    logging.info('Policy Config:')
    pprint.pprint(policy_config)

    return env_config, policy_config


def get_dataset_iterator(path,
                         tf_agent,
                         batch_size,
                         prefetch_capacity,
                         num_parallel_calls=8,
                         episodes_per_file=500,
                         sampling_dataset_timesteps_per_episode_hint=8):
    """Get the dataset iterator.

    Args:
        path: The path to the dataset.
        tf_agent: The Tensorflow agent.
        batch_size: Batch size for training.
        prefetch_capacity: Capacity of prefetching data.
        num_parallel_calls: Number of parallel calls for reading data.
        episodes_per_file: Number of stored episodes in each TFRecord file.
        sampling_dataset_timesteps_per_episode_hint: Constant for memory size.

    Returns:
        dataset_iterator: The dataset iterator.
    """
    # Prepare replay buffer as dataset with invalid transitions filtered.
    def _filter_invalid_transition(trajectories):
        return ~tf.reduce_any(trajectories.is_boundary()[:-1])

    replay_buffer = tfrecord_replay_buffer.TFRecordReplayBuffer(
        data_spec=tf_agent.collect_data_spec,
        experiment_id='exp',
        file_prefix=os.path.join(path, 'data'),
        episodes_per_file=episodes_per_file,
        sampling_dataset_timesteps_per_episode_hint=(
            sampling_dataset_timesteps_per_episode_hint))
    dataset = replay_buffer.as_dataset(
        sample_batch_size=5 * batch_size,
        num_steps=tf_agent.train_sequence_length,
        num_parallel_calls=num_parallel_calls).apply(
            tf.data.experimental.unbatch()).filter(
                _filter_invalid_transition).batch(batch_size).prefetch(
                        buffer_size=prefetch_capacity).repeat()
    dataset_iterator = tf.compat.v1.data.make_initializable_iterator(
        dataset)
    return dataset_iterator


def train_eval(tf_env,
               tf_agent_ctor,
               policy_config,
               root_dir,
               rb_dir,
               # Training.
               num_iterations=5000000,
               batch_size=64,
               learning_rate=1e-4,
               gradient_clipping=25.0,
               # Evaluation.
               eval_batch_size=512,
               eval_interval=5000,
               # Summary.
               train_checkpoint_interval=10000,
               log_interval=1000,
               summary_interval=1000,
               summaries_flush_secs=10,
               debug_summaries=True,
               summarize_grads_and_vars=True):
    """A simple train and eval function."""
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_summary_flush_op = eval_summary_writer.flush()

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):

        # Get the data specs from the environment
        time_step_spec = tf_env.time_step_spec()
        action_spec = tf_env.action_spec()

        logging.info('learning_rate: %f', learning_rate)

        tf_agent = tf_agent_ctor(
                time_step_spec,
                action_spec,
                policy_config=policy_config,
                learning_rate=learning_rate,
                gradient_clipping=gradient_clipping,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars,
                train_step_counter=global_step)

        # Training data.
        train_dataset_iterator = get_dataset_iterator(
            path=os.path.join(rb_dir, 'train'),
            tf_agent=tf_agent,
            batch_size=batch_size,
            prefetch_capacity=20 * batch_size)
        train_trajectories = train_dataset_iterator.get_next()
        train_op = tf_agent.train(train_trajectories)

        # Evaluation data.
        eval_dataset_iterator = get_dataset_iterator(
            path=os.path.join(rb_dir, 'eval'),
            tf_agent=tf_agent,
            batch_size=batch_size,
            prefetch_capacity=5 * batch_size)
        eval_trajectories = eval_dataset_iterator.get_next()

        with eval_summary_writer.as_default():
            eval_op = tf_agent.eval(eval_trajectories)
            eval_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()

        var_list = framework.get_variables()
        saver = tf.train.Saver(var_list=var_list, name='saver')

        with tf.compat.v1.Session() as sess:
            # Initialize training.
            sess.run(train_dataset_iterator.initializer)
            sess.run(eval_dataset_iterator.initializer)
            common.initialize_uninitialized_variables(sess)

            sess.run(train_summary_writer.init())
            sess.run(eval_summary_writer.init())

            global_step_call = sess.make_callable(global_step)
            train_step_call = sess.make_callable(train_op)
            eval_step_call = sess.make_callable([eval_op, eval_summary_ops])

            global_step_val = sess.run(global_step)

            timed_at_step = global_step_call()
            time_acc = 0
            steps_per_second_ph = tf.compat.v1.placeholder(
                    tf.float32, shape=(), name='steps_per_sec_ph')
            steps_per_second_summary = tf.compat.v2.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_second_ph,
                    step=global_step)

            for _ in range(num_iterations):
                # Evaluation.
                if global_step_val % eval_interval == 0:
                    total_eval_loss, _ = eval_step_call()
                    logging.info('Evaluation: step = %d, loss = %f',
                                 global_step_val,
                                 total_eval_loss.loss)
                    sess.run(eval_summary_flush_op)

                # Training.
                start_time = time.time()
                total_loss = train_step_call()
                time_acc += time.time() - start_time
                global_step_val = global_step_call()
                if global_step_val % log_interval == 0:
                    logging.info('step = %d, loss = %f',
                                 global_step_val,
                                 total_loss.loss)
                    steps_per_sec = (global_step_val - timed_at_step
                                     ) / time_acc
                    logging.info('%.3f steps/sec', steps_per_sec)
                    sess.run(steps_per_second_summary,
                             feed_dict={steps_per_second_ph: steps_per_sec})
                    timed_at_step = global_step_val
                    time_acc = 0

                # Saving checkpoints.
                if global_step_val % train_checkpoint_interval == 0:
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    logging.info('Saving model at step %d to %s ...',
                                 global_step_val, checkpoint_path)
                    saver.save(sess, checkpoint_path)


def main():
    args = parse_args()

    # Set the random seed.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # Configuration.
    env_config, policy_config = parse_config_files_and_bindings(args)

    # Create the environment.
    simulator = Simulator(assets_dir=args.assets_dir)

    # Create the environment.
    tf_env = tf_py_environment.TFPyEnvironment(
        suite_env.load(args.env, config=env_config, simulator=simulator))

    # Train and evaluate.
    tf.compat.v1.enable_resource_variables()
    tf_agent_ctor = AGENT_CTORS[args.agent]
    train_eval(tf_env=tf_env,
               tf_agent_ctor=tf_agent_ctor,
               policy_config=policy_config,
               root_dir=args.working_dir,
               rb_dir=args.rb_dir,
               learning_rate=args.learning_rate,
               debug_summaries=args.debug,
               summarize_grads_and_vars=args.debug)


if __name__ == '__main__':
    main()
