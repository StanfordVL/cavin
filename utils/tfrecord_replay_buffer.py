"""A replay buffer of nests of Tensors, backed by TFRecords files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import threading
import time
import uuid

from absl import logging
from six.moves import queue as Queue
import tensorflow as tf
from tf_agents.replay_buffers import tfrecord_replay_buffer
from tf_agents.trajectories import time_step

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.data.util import nest as data_nest  # TF internal
from tensorflow.python.ops.distributions import util as distributions_util  # TF internal
# pylint:enable=g-direct-tensorflow-import


StepType = time_step.StepType

FILE_FORMAT = (
    '{file_prefix}_{experiment_id}_'
    '{YYYY:04d}{MM:02d}{DD:02d}_'
    '{hh:02d}{mm:02d}{ss:02d}_'
    '{hash}.tfrecord')


BufferInfo = collections.namedtuple('BufferInfo', ['ids'])


class _Stop(object):
  pass


class _Flush(object):

  def __init__(self, lock):
    self.lock = lock
    self.condition_var = threading.Condition(lock=lock)


# Signal object for writers to wrap up, flush their queues, and exit.
_STOP = _Stop()

# The buffer size of values to put on a write queue at a time.
# Tuned via benchmarks in tfrecord_replay_buffer_test.
_QUEUE_CHUNK_SIZE = 16


class TFRecordReplayBuffer(tfrecord_replay_buffer.TFRecordReplayBuffer):
  """A replay buffer that stores data in TFRecords.
  """

  def __init__(self,
               experiment_id,
               data_spec,
               file_prefix,
               episodes_per_file,
               time_steps_per_file=None,
               record_options=None,
               sampling_dataset_timesteps_per_episode_hint=256,
               dataset_block_keep_prob=1.0,
               dataset_batch_drop_remainder=True,
               seed=None):
    """Creates a TFRecordReplayBuffer.
    """
    super(TFRecordReplayBuffer, self).__init__(
        experiment_id=experiment_id,
        data_spec=data_spec,
        file_prefix=file_prefix,
        episodes_per_file=episodes_per_file,
        time_steps_per_file=time_steps_per_file,
        record_options=record_options,
        sampling_dataset_timesteps_per_episode_hint=(
            sampling_dataset_timesteps_per_episode_hint),
        dataset_block_keep_prob=dataset_block_keep_prob,
        dataset_batch_drop_remainder=dataset_batch_drop_remainder,
        seed=seed)

  def _as_dataset(self,
                  sample_batch_size=None,
                  num_steps=None,
                  num_parallel_calls=None):
    """Creates a dataset that returns entries from the buffer.
    """
    if num_steps is None:
      if sample_batch_size is not None:
        raise ValueError(
            'When num_steps is None, sample_batch_size must be '
            'None but saw: %s' % (sample_batch_size,))
    else:
      if sample_batch_size is None or sample_batch_size <= 0:
        raise ValueError(
            'When num_steps is not None, sample_batch_size must be '
            'an integer > 0, saw: %s' % (sample_batch_size,))

    # data_tf.nest.flatten does not flatten python lists, tf.nest.flatten does.
    flat_data_spec = tf.nest.flatten(self._data_spec)
    if flat_data_spec != data_nest.flatten(self._data_spec):
      raise ValueError(
          'Cannot perform gather; data spec contains lists and this conflicts '
          'with gathering operator.  Convert any lists to tuples.  '
          'For example, if your spec looks like [a, b, c], '
          'change it to (a, b, c).  Spec structure is:\n  {}'.format(
              tf.nest.map_structure(lambda spec: spec.dtype, self._data_spec)))

    filename_seed = distributions_util.gen_new_seed(self._data.seed,
                                                    salt='filename_seed')

    batch_seed = distributions_util.gen_new_seed(self._data.seed,
                                                 salt='batch_seed')

    drop_block_seed = distributions_util.gen_new_seed(self._data.seed,
                                                      salt='drop_block')

    # TODO(b/128998627): Use a different seed for each file by mapping a count
    # with the filename and doing the seed generation in graph mode.
    per_episode_seed = distributions_util.gen_new_seed(self._data.seed,
                                                       salt='per_episode_seed')

    block_keep_prob = self._data.dataset_block_keep_prob
    dropping_blocks = (tf.is_tensor(block_keep_prob) or block_keep_prob != 1.0)
    if dropping_blocks:
      # empty_block_ds is in format (is_real_data=False, empty_data)
      empty_block_ds = tf.data.Dataset.from_tensors(
          (False, tf.fill([num_steps], '')))

      def select_true_or_empty(_):
        # When this returns 0, select the true block.  When this returns 1,
        # select the empty block.
        return tf.cast(
            tf.random.uniform((), seed=drop_block_seed) > block_keep_prob,
            tf.int64)

      true_or_empty_block_selector_ds = (
          tf.data.experimental.Counter().map(select_true_or_empty))

    def list_and_shuffle_files(_):
      filenames = tf.io.matching_files(
          tf.strings.join(
              (self._data.file_prefix, self._data.experiment_id, '*'),
              separator='_'))
      shuffled = tf.random.shuffle(filenames, seed=filename_seed)
      return shuffled

    def parse_blocks_from_record(records):
      """Decode `FeatureList` tensor `records`.

      Args:
        records: `tf.string` tensor of shape either `[]` or `[batch_size]`.

      Outputs:
        A struct matching `self._data_spec` containing tensors.
        If `num_steps is not None`, it contains tensors with shape
        `[batch_size, num_steps, ...]`; otherwise they have shape `[...]`.
      """
      # If `num_steps is None`, then:
      #  records is shaped [].
      #  features is shaped [len(flatten(self._data_spec))].
      # otherwise:
      #  records is shaped [batch_size].
      #  features is shaped [batch_size, len(flatten(self._data_spec))].
      _, features = tf.io.decode_proto(
          bytes=records,
          message_type='tensorflow.FeatureList',
          field_names=['feature'],
          output_types=[tf.string])
      features = features.pop()
      num_features = len(flat_data_spec)
      features = tf.unstack(features, num_features, axis=-1)
      decoded_features = []
      for feature, spec in zip(features, flat_data_spec):
        decoded_feature = _decode_feature(
            feature,
            spec,
            has_outer_dims=num_steps is not None)
        decoded_features.append(decoded_feature)
      return tf.nest.pack_sequence_as(self._data_spec, decoded_features)

    def read_and_block_fixed_length_tfrecord_file(filename):
      """Read records from `filename`, window them into fixed len blocks.

      This function also optionally subsamples and shuffles the blocks.

      Windowed records from filename come as a stream and prior to subsampling
      and shuffling, the stream contains blocks of the form:

         [r0, r1, ..., r_{num_steps - 1}]
         [r1, r2, ..., r_{num_steps}]
         [r2, r3, ..., r_{num_steps + 1}]
         ...

      Args:
        filename: A scalar string `Tensor` with the TFRecord filename.

      Returns:
        A `tf.data.Dataset` instance.
      """
      def drop_or_batch_window(ds):
        if not dropping_blocks:
          return ds.batch(num_steps, drop_remainder=True)
        else:
          # batched_ds is in format (is_real_data=True, true_ds)
          batched_ds = tf.data.Dataset.zip(
              (tf.data.Dataset.from_tensors(True),
               ds.batch(num_steps, drop_remainder=True)))
          return (
              tf.data.experimental.choose_from_datasets(
                  (batched_ds, empty_block_ds),
                  true_or_empty_block_selector_ds)
              .take(1)
              .filter(lambda is_real_data, _: is_real_data)
              .map(lambda _, true_block: true_block))
      return (
          tf.data.TFRecordDataset(
              filename,
              compression_type=_compression_type_string(
                  self._data.record_options))
          .window(num_steps, shift=1, stride=1, drop_remainder=True)
          .flat_map(drop_or_batch_window)
          .shuffle(buffer_size=self._data.per_file_shuffle_buffer_size,
                   seed=per_episode_seed))

    def read_and_block_variable_length_tfrecord_file(filename):
      """Read records from `filename`, window them into variable len blocks."""
      def create_ta(spec):
        return tf.TensorArray(
            size=0, dynamic_size=True, element_shape=spec.shape,
            dtype=spec.dtype)
      empty_tas = tf.nest.map_structure(create_ta, self._data_spec)

      def parse_and_block_on_episode_boundaries(partial_tas, record):
        frame = parse_blocks_from_record(record)
        updated_tas = tf.nest.map_structure(
            lambda ta, f: ta.write(ta.size(), f),
            partial_tas, frame)
        # If we see a LAST field, then emit empty TAs for the state and updated
        # TAs for the output.  Otherwise emit updated TAs for the state and
        # empty TAs for the output (the empty output TAs will be filtered).
        return tf.cond(
            tf.equal(frame.step_type, StepType.LAST),
            lambda: (empty_tas, updated_tas),
            lambda: (updated_tas, empty_tas))

      stack_tas = lambda tas: tf.nest.map_structure(lambda ta: ta.stack(), tas)
      remove_intermediate_arrays = lambda tas: tas.step_type.size() > 0

      return (
          tf.data.TFRecordDataset(
              filename,
              compression_type=_compression_type_string(
                  self._data.record_options))
          .apply(
              tf.data.experimental.scan(empty_tas,
                                        parse_and_block_on_episode_boundaries))
          .filter(remove_intermediate_arrays)
          .map(stack_tas)
          .shuffle(buffer_size=self._data.per_file_shuffle_buffer_size,
                   seed=per_episode_seed))

    interleave_shuffle_buffer_size = (
        (num_parallel_calls or sample_batch_size or 4)
        * self._data.sampling_dataset_timesteps_per_episode_hint)

    cycle_length = max(num_parallel_calls or sample_batch_size or 0, 4)

    if num_steps is None:
      read_and_block_fn = read_and_block_variable_length_tfrecord_file
    else:
      read_and_block_fn = read_and_block_fixed_length_tfrecord_file

    # Use tf.data.Dataset.from_tensors(0).map(...) to call the map() code once
    # per initialization.  This means that when the iterator is reinitialized,
    # we get a new list of files.
    ds = (tf.data.Dataset.from_tensors(0)
          .map(list_and_shuffle_files)
          .flat_map(tf.data.Dataset.from_tensor_slices)
          # Interleave between blocks of records from different files.
          .interleave(read_and_block_fn,
                      cycle_length=cycle_length,
                      block_length=self._data.per_file_shuffle_buffer_size,
                      num_parallel_calls=(num_parallel_calls
                                          or tf.data.experimental.AUTOTUNE))
          .shuffle(
              buffer_size=interleave_shuffle_buffer_size,
              seed=batch_seed))

    # Batch and parse the blocks.  If `num_steps is None`, parsing has already
    # happened and we're not batching.
    if num_steps is not None:
      ds = (ds.batch(batch_size=sample_batch_size,
                     drop_remainder=self._data.drop_remainder)
            .map(parse_blocks_from_record))

    return ds


def _generate_filename(file_prefix, experiment_id):
  now = time.gmtime()
  return FILE_FORMAT.format(
      file_prefix=file_prefix,
      experiment_id=experiment_id,
      YYYY=now.tm_year,
      MM=now.tm_mon,
      DD=now.tm_mday,
      hh=now.tm_hour,
      mm=now.tm_min,
      ss=now.tm_sec,
      hash=uuid.uuid1())


def _create_send_batch_py(rb_data):
  """Return a function to send data to writer queues.

  This functor returns a function

    send_batch_py(step_type, flat_serialized) -> True

  which (possibly) initializes and sends data to writer queues pointed to
  by `rb_data`.

  This function takes an `_RBData` instance instead of being a method of
  `TFRecordReplayBuffer` because we want to avoid an extra dependency
  on the replay buffer object from within TensorFlow's py_function.  This
  circular dependency occurs because of a cell capture from the TF runtime
  on `self` if `self` is ever used within a py_function.

  Args:
    rb_data: An instance of `_RBData`.

  Returns:
    A function `send_batch_py(step_type, flat_serialized) -> True`.
  """
  def send_batch_py(step_type, flat_serialized):
    """py_function that sends data to a writer thread."""
    # NOTE(ebrevdo): Here we have a closure over ONLY rb._data, not over rb
    # itself.  This avoids the circular dependency.
    step_type = step_type.numpy()
    flat_serialized = flat_serialized.numpy()
    batch_size = step_type.shape[0]
    _maybe_initialize_writers(batch_size, rb_data)
    for batch in range(batch_size):
      queue_buffer = rb_data.queue_buffers[batch]
      queue_buffer.append(
          (rb_data.experiment_id, step_type[batch], flat_serialized[batch]))
      if len(queue_buffer) >= rb_data.queue_chunk_size:
        rb_data.queues[batch].put(list(queue_buffer))
        del queue_buffer[:]
    return True
  return send_batch_py


def _maybe_initialize_writers(batch_size, rb_data):
  """Initialize the queues and threads in `rb_data` given `batch_size`."""
  with rb_data.lock:
    if rb_data.batch_size is None:
      rb_data.batch_size = batch_size
      if batch_size > 64:
        logging.warning(
            'Using a batch size = %d > 64 when writing to the '
            'TFRecordReplayBuffer, which can cause python thread contention '
            'and impact performance.')
    if batch_size != rb_data.batch_size:
      raise ValueError(
          'Batch size does not match previous batch size: %d vs. %d'
          % (batch_size, rb_data.batch_size))
    if not rb_data.writer_threads:
      rb_data.queue_buffers.extend([list() for _ in range(batch_size)])
      rb_data.queues.extend([Queue.Queue() for _ in range(batch_size)])
      # pylint: disable=g-complex-comprehension
      rb_data.writer_threads.extend([
          threading.Thread(
              target=_process_write_queue,
              name='process_write_queue_%d' % i,
              kwargs={
                  'queue': rb_data.queues[i],
                  'episodes_per_file': rb_data.episodes_per_file,
                  'time_steps_per_file': rb_data.time_steps_per_file,
                  'file_prefix': rb_data.file_prefix,
                  'record_options': rb_data.record_options
              })
          for i in range(batch_size)
      ])
      # pylint: enable=g-complex-comprehension
      for thread in rb_data.writer_threads:
        thread.start()


def _process_write_queue(queue, episodes_per_file, time_steps_per_file,
                         file_prefix, record_options):
  """Process running in a separate thread that writes the TFRecord files."""
  writer = None
  num_steps = 0
  num_episodes = 0

  while True:
    item = queue.get()
    if item is _STOP:
      if writer is not None:
        try:
          writer.close()
        except tf.errors.OpError as e:
          logging.error(str(e))
      return
    elif isinstance(item, _Flush):
      if writer is not None:
        try:
          writer.flush()
        except tf.errors.OpError as e:
          logging.error(str(e))
      with item.lock:
        item.condition_var.notify()
      continue

    for experiment_id, step_type, serialized_feature_list in item:
      num_steps += 1
      if step_type == StepType.FIRST:
        num_episodes += 1
      if (writer is None
          or (step_type == StepType.FIRST
              and num_episodes >= episodes_per_file)
          or (time_steps_per_file is not None
              and num_steps >= time_steps_per_file)):
        filename = _generate_filename(
            file_prefix=file_prefix, experiment_id=experiment_id)
        if writer is None:
          try:
            tf.io.gfile.makedirs(filename[:filename.rfind('/')])
          except tf.errors.OpError as e:
            logging.error(str(e))
        else:
          try:
            writer.close()
          except (AttributeError, tf.errors.OpError) as e:
            logging.error(str(e))
        num_episodes = 0
        num_steps = 0
        try:
          writer = tf.io.TFRecordWriter(filename, record_options)
        except tf.errors.OpError as e:
          logging.error(str(e))

      try:
        writer.write(serialized_feature_list)
      except (tf.errors.OpError, AttributeError) as e:
        logging.error(str(e))


def _encode_to_feature(t):
  """Encodes batched tensor `t` to a batched `tensorflow.Feature` tensor string.

  - Integer tensors are encoded as `int64_list`.
  - Floating point tensors are encoded as `float_list`.
  - String tensors are encoded as `bytes_list`.

  Args:
    t: A `tf.Tensor` shaped `[batch_size, ...]`.

  Returns:
    A string `tf.Tensor` with shape `[batch_size]` containing serialized
    `tensorflow.Feature` proto.

  Raises:
    NotImplementedError: If `t.dtype` is not a supported encoding dtype.
  """
  batch_size = tf.compat.dimension_at_index(t.shape, 0) or tf.shape(t)[0]
  t = tf.reshape(t, [batch_size, -1])
  num_elements_per_batch = (
      tf.compat.dimension_at_index(t.shape, 1) or tf.shape(t)[1])
  # TODO(b/129368627): Revisit writing everything as a BytesList once we have
  # support for getting string-like views of dense tensors.
  if t.dtype.is_integer or t.dtype.base_dtype == tf.bool:
    field_name = 'int64_list'
    message_type = 'tensorflow.Int64List'
    t = tf.cast(t, tf.int64)
  elif t.dtype.is_floating:
    field_name = 'float_list'
    message_type = 'tensorflow.FloatList'
    t = tf.cast(t, tf.float32)  # We may lose precision here
  elif t.dtype.base_dtype == tf.string:
    field_name = 'bytes_list'
    message_type = 'tensorflow.BytesList'
  else:
    raise NotImplementedError('Encoding tensor type %d is is not supported.'
                              % t.dtype)
  batch_size = tf.shape(t)[0]
  values_list = tf.io.encode_proto(
      sizes=tf.fill([batch_size, 1], num_elements_per_batch),
      values=[t],
      field_names=['value'],
      message_type=message_type)
  return tf.io.encode_proto(
      sizes=tf.fill([batch_size, 1], 1),
      values=[tf.expand_dims(values_list, 1)],
      field_names=[field_name],
      message_type='tensorflow.Feature')


def _decode_feature(feature, spec, has_outer_dims):
  """Decodes batched serialized `tensorflow.Feature` to a batched `spec` tensor.

  - Integer tensors are decoded from `int64_list`.
  - Floating point tensors are decoded from `float_list`.
  - String tensors are decoded `bytes_list`.

  Args:
    feature: A `tf.Tensor` of type string shaped `[batch_size, num_steps]`.
    spec: A `tf.TensorSpec`.
    has_outer_dims: Python bool, whether the feature has a batch dim or not.

  Returns:
    Returns `tf.Tensor` with shape `[batch_size, num_steps] + spec.shape`
    having type `spec.dtype`.  If `not has_outer_dims`, then the tensor has no
    `[batch_size, num_steps]` prefix.

  Raises:
    NotImplementedError: If `spec.dtype` is not a supported decoding dtype.
  """
  if has_outer_dims:
    feature.shape.assert_has_rank(2)
  else:
    feature.shape.assert_has_rank(0)

  # This function assumes features come in as encoded tensorflow.Feature strings
  # with shape [batch, num_steps].
  if spec.dtype.is_integer or spec.dtype.base_dtype == tf.bool:
    field_name = 'int64_list'
    message_type = 'tensorflow.Int64List'
    feature_dtype = tf.int64
  elif spec.dtype.is_floating:
    field_name = 'float_list'
    message_type = 'tensorflow.FloatList'
    feature_dtype = tf.float32
  elif spec.dtype.base_dtype == tf.string:
    field_name = 'bytes_list'
    message_type = 'tensorflow.BytesList'
    feature_dtype = tf.string
  else:
    raise NotImplementedError('Decoding spec type %d is is not supported.'
                              % spec.dtype)

  _, value_message = tf.io.decode_proto(
      bytes=feature,
      message_type='tensorflow.Feature',
      field_names=[field_name],
      output_types=[tf.string])
  value_message = value_message.pop()
  value_message = tf.squeeze(value_message, axis=-1)

  _, values = tf.io.decode_proto(
      bytes=value_message,
      message_type=message_type,
      field_names=['value'],
      output_types=[feature_dtype])
  values = values.pop()
  values_shape = tf.shape(values)
  if has_outer_dims:
    batch_size = tf.compat.dimension_value(values.shape[0]) or values_shape[0]
    num_steps = tf.compat.dimension_value(values.shape[1]) or values_shape[1]
    values = tf.reshape(
        values, [batch_size, num_steps] + spec.shape.as_list())
  else:
    values = tf.reshape(values, spec.shape.as_list())
  return tf.cast(values, spec.dtype)


class WriterCleanup(object):
  """WriterCleanup class.

  When garbage collected, this class flushes all queue buffers and writer
  threads.
  """

  def __init__(self, rb_data):
    self._data = rb_data

  def __del__(self):
    # NOTE(ebrevdo): This may spew a lot of noise at program shutdown due to
    # random module unloading.  If it does, it's safe to ignore but we might
    # consider wrapping it inside a broad try: ... except:, to avoid the noise.
    if not self._data.writer_threads:
      return
    for queue, buf in zip(self._data.queues, self._data.queue_buffers):
      if buf:
        queue.put(list(buf))
        del buf[:]
      queue.put(_STOP)
    for t in self._data.writer_threads:
      t.join()


def _compression_type_string(record_options):
  if record_options is None:
    return None
  assert isinstance(record_options, tf.io.TFRecordOptions), type(record_options)
  return record_options.compression_type
