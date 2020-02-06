#!/usr/bin/env python

"""Filter corrupted tfrecords.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import glob

import tensorflow as tf


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data',
        dest='data_dir',
        type=str,
        help='Path to the training set.',
        required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    data_pattern = os.path.join(args.data_dir, '*.tfrecord')
    data_paths = glob.glob(data_pattern)

    print('Checking files...')

    for data_path in data_paths:
        print(data_path)
        try:
            for example in tf.python_io.tf_record_iterator(data_path):
                pass
        except Exception:
            print('Remove %s' % (data_path))
            os.remove(data_path)


if __name__ == '__main__':
    main()
