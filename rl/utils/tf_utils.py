import abc
import contextlib
import functools
import logging

import numpy as np
import tensorflow as tf


def lazy_function(function):
    attribute = '_cache_' + function.__name__
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

def flatten(t):
  return tf.reshape(t, [-1, np.prod(t.shape[1:]).value])

def orthogonal_initializer(scale=1.0):
  # taken from https://github.com/openai/baselines/tree/master/baselines/ppo2
  def _initializer(shape, dtype, partition_info=None):
    shape = tuple(shape)
    if len(shape) == 2:
      flat_shape = shape
    elif len(shape) == 4: # assumes NHWC
      flat_shape = (np.prod(shape[:-1]), shape[-1])
    else:
      raise NotImplementedError()
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


@contextlib.contextmanager
def variable_and_name_scopes(scope):
  """ Restores both variable and name scopes.

    Both scopes might need to be restored to allow continuation of building
    process. Restoring only variable scope might lead to the creation of
    new name scope which is not desirable in some cases.
  """
  # Manually reopen name_scope inside variable_scope of
  # the algorithm. See https://github.com/tensorflow/tensorflow/issues/6189.
  # This is also the way it is done in tf.layers.base.
  with tf.variable_scope(scope) as restored,\
      tf.name_scope(scope.original_name_scope):
    yield restored

def scoped(func):
  """ Decorator that adds variable and name scope for function of a class.

  The class need to have `scope` property of type `tf.VariableScope`. This
  variable scope will be restored together with the underlying name scope.
  """
  def _scoped(self, *args, **kwargs):
    with variable_and_name_scopes(self.scope):
      return func(self, *args, **kwargs)
  return _scoped


class NetworkStructure(abc.ABC):
  """ Conceptually separate structure in tensorflow graph. """
  def __new__(cls, *args, before_build_hooks=None, after_build_hooks=None,
              name=None, **kwargs):
    # We use __new__ since we want the env author to be able to
    # override __init__ without remembering to call super.
    network_structure = super(NetworkStructure, cls).__new__(cls)
    network_structure._is_built = False
    network_structure._name = name or cls.__name__
    network_structure._scope = next(
        tf.variable_scope(network_structure._name).gen)
    network_structure._before_build_hooks = before_build_hooks or []
    network_structure._after_build_hooks = after_build_hooks or []
    return network_structure

  @property
  def scope(self):
    return self._scope

  @property
  def is_built(self):
    return self._is_built

  def add_before_build_hook(self, hook):
    self._before_build_hooks.append(hook)

  def add_after_build_hook(self, hook):
    self._after_build_hooks.append(hook)

  @scoped
  def build(self, *args, **kwargs):
    if not self.is_built:
      logging.info("Building {}".format(self._name))
      for hook in self._before_build_hooks:
        hook(self, *args, **kwargs)
      self._build(*args, **kwargs)
      self._is_built = True
      for hook in self._after_build_hooks:
        hook(self, *args, **kwargs)

  @abc.abstractmethod
  def _build(self):
    ...
