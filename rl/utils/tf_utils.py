import abc
from collections import defaultdict
import functools
from logging import getLogger

import numpy as np
import tensorflow as tf

logger = getLogger("rl")


def lazy_function(function):
    attribute = '_cache_' + function.__name__

    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def purge_orphaned_summaries(summary_writer, step):
  summary_writer.add_session_log(
    tf.SessionLog(status=tf.SessionLog.START), step)


def read_events(event_filename, data=None, purge_orphaned=True):
  data = data if data is not None else defaultdict(dict)
  for e in tf.train.summary_iterator(event_filename):
    if purge_orphaned and e.session_log.status == tf.SessionLog.START:
      for tag in data.keys():
        data[tag] = {
            step_time: val
            for step_time, val in data[tag].items()
            if step_time[0] < e.step
        }

    for v in e.summary.value:
      data[v.tag][(e.step, e.wall_time)] = v.simple_value
  return data


def orthogonal_initializer(scale=1.0):
  # taken from https://github.com/openai/baselines/tree/master/baselines/ppo2
  def _initializer(shape, dtype, partition_info=None):
    shape = tuple(shape)
    if len(shape) == 2:
      flat_shape = shape
    elif len(shape) == 4:  # assumes NHWC
      flat_shape = (np.prod(shape[:-1]), shape[-1])
    else:
      raise NotImplementedError("Not supported shape: {}".format(shape))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
  return _initializer


def normalized_columns_initializer(stddev=1.0):
  def _initializer(shape, dtype=None, partition_info=None):
    out = np.random.randn(*shape).astype(np.float32)
    out *= stddev / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)
  return _initializer


def torch_default_initializer():
  """Default initializer in torch.

   The initializer is similar to the He uniform initializer as
   described in  http://arxiv.org/abs/1502.01852, but uses
   different value in numerator.

   https://github.com/torch/nn/blob/master/Linear.lua#L21
   https://github.com/torch/nn/blob/master/SpatialConvolution.lua#L34
  """
  def _initializer(shape, dtype=None, partition_info=None):
    shape = tuple(shape)
    fan_in = np.prod(shape[:-1])
    stddev = 1. / np.sqrt(fan_in)
    return tf.random_uniform(shape, minval=-stddev, maxval=stddev, dtype=dtype)
  return _initializer


def partial_restore(variables, checkpoint, session=None, scope=None):
  meta_file = checkpoint + ".meta"
  with tf.Graph().as_default(), tf.Session() as sess, sess.as_default():
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, checkpoint)
    checkpoint_vars = {
        v.name: v.eval()
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=scope)
    }
  for v in variables:
    v.load(checkpoint_vars[v.name])


def explained_variance(targets, predictions):
  targets, predictions = tf.squeeze(targets), tf.squeeze(predictions)
  tf.assert_rank_in(targets, [0, 1])
  tf.assert_rank_in(predictions, [0, 1])
  var_targets = tf.cond(
      tf.equal(tf.rank(targets), 0),
      lambda: tf.constant(0, dtype=tf.float32),
      lambda: tf.nn.moments(targets, axes=[0])[1]
  )
  return tf.cond(
      tf.equal(var_targets, 0),
      lambda: tf.constant(np.nan),
      lambda: (1
               - tf.nn.moments(targets - predictions, axes=[0])[1]
               / var_targets)
  )


def huber_loss(x, delta=1.0):
  with tf.variable_scope("huber_loss"):
    abs_x = tf.abs(x)
    return tf.where(
        abs_x < delta,
        0.5 * tf.square(x),
        delta * (abs_x - 0.5 * delta)
    )


class BuildInterface(abc.ABC):
  """ Interface for building parts of tensorflow graphs. """
  def __init__(self, name=None):
    self._is_built = False
    self._name = name or self.__class__.__name__
    with tf.variable_scope(self._name) as captured_scope:
      self._scope = captured_scope
    self._before_build_hooks = []
    self._after_build_hooks = []

  @property
  def is_built(self):
    return self._is_built

  @property
  def scope(self):
    return self._scope

  def add_before_build_hook(self, hook):
    self._before_build_hooks.append(hook)

  def add_after_build_hook(self, hook):
    self._after_build_hooks.append(hook)

  def build(self, *args, **kwargs):
    with tf.variable_scope(self._scope),\
        tf.name_scope(self.scope.original_name_scope):
      if not self.is_built:
        logger.info("Building {}".format(self._name))
        for hook in self._before_build_hooks:
          hook(self, *args, **kwargs)
        self._build(*args, **kwargs)
        self._is_built = True
        for hook in self._after_build_hooks:
          hook(self, *args, **kwargs)

  @abc.abstractmethod
  def _build(self):
    ...
