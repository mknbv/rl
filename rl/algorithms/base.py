import abc

import tensorflow as tf

import rl.utils.tf_utils as tfu


__all__ = ["BaseAlgorithm"]


class BaseAlgorithm(tfu.NetworkStructure):
  def __init__(self, global_policy, local_policy, name=None):
    if local_policy is not None and\
        type(global_policy) is not type(local_policy):
      raise TypeError(
      "`global_policy` and `local_policy` must be of the same type")
    super(BaseAlgorithm, self).__init__(name=name)
    self._global_policy = global_policy
    self._local_policy = local_policy
    self._loss = None
    self._summaries = None
    self._train_op = None
    self._sync_op = None

  @property
  def local_policy(self):
    return self._local_policy

  @property
  def global_policy(self):
    return self._global_policy

  @property
  def local_or_global_policy(self):
    return self._local_policy or self._global_policy

  @property
  @abc.abstractmethod
  def logging_fetches(self):
    ...

  @property
  def loss(self):
    return self._loss

  @property
  def train_op(self):
    return self._train_op

  @property
  def summaries(self):
    return self._summaries

  @property
  def sync_op(self):
    return self._sync_op

  @tfu.scoped
  def build_loss(self):
    self._loss = self._build_loss()

  @tfu.scoped
  def build_train_op(self, optimizer):
    self._train_op = self._build_train_op(optimizer)

  @tfu.scoped
  def build_summaries(self):
    self._summaries = self._build_summaries()

  @tfu.scoped
  def build_sync_op(self):
    self._sync_op = self._build_sync_op()

  def get_feed_dict(self, sess, summary_time=False):
    return self._get_feed_dict(sess, summary_time=summary_time)

  def _build(self, optimizer, worker_device=None, device_setter=None):
    """ Adds all the variables and ops needed by this algorithm.

    This currently does not support partially build algorithms, so
    if, for example, `build_loss` was called before calling this
    function, it might fail in weird ways.  `worker_device` and
    `device_setter` are required when algorithm implements
    `local_policy` (`algorithm.local_policy is not None`). When
    `local_policy` is not implemented only worker_device is used.

    Args:
      optimizer: optimizer for the algorithm.
      worker_device: device which will be used to build `local_policy` and
        the algorithm itself.
      device_setter: device setter function for `global_policy`.
    """
    if self.local_policy is not None and device_setter is None:
      raise TypeError(
          "{} implements both local and global policies, therefore"
          "worker_device and device_setter must be provided"
          .format(self.__class__.__name__)
      )
    if self.local_policy is None:
      device_setter = worker_device
    with tf.device(device_setter):
      self.global_policy.build()
    with tf.device(worker_device):
      if self.local_policy is not None:
        self._local_policy.build()
      self.build_loss()
      self.build_train_op(optimizer)
      self.build_summaries()
      self.build_sync_op()

  @abc.abstractmethod
  def _build_loss(self):
    ...

  @abc.abstractmethod
  def _build_summaries(self):
    ...

  def _build_sync_op(self):
    if self.local_policy is not None:
      return tf.group(*[
          v1.assign(v2) for v1, v2 in zip(self._local_policy.var_list(),
                                          self._global_policy.var_list())
      ])
    else:
      return tf.no_op()

  @abc.abstractmethod
  def _build_train_op(self):
    ...

  @abc.abstractmethod
  def _get_feed_dict(self, sess, summary_time=False):
    ...
