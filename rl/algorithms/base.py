import abc
import tensorflow as tf

import rl.utils.tf_utils as tfu


# TODO: implement some way of keeping track of the build stage and build
# dependencies.
class BaseAlgorithm(tfu.NetworkStructure):
  def __init__(self, global_policy, local_policy, name=None):
    if local_policy is not None and\
        type(global_policy) is not type(local_policy):
      raise TypeError(
      "`global_policy` and `local_policy` must be of the same type")
    self._global_policy = global_policy
    self._local_policy = local_policy
    self._loss = None
    self._summaries = None
    self._train_op = None
    self._sync_ops = None

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
  def sync_ops(self):
    return self._sync_ops

  @tfu.scoped
  def build_loss(self, worker_device=None, device_setter=None):
    self._loss = self._build_loss(worker_device=worker_device,
                                  device_setter=device_setter)

  @tfu.scoped
  def build_train_op(self, optimizer, worker_device=None, device_setter=None):
    batch_size = tf.shape(self.local_or_global_policy.inputs)[0]
    batch_size = tf.to_int64(batch_size)
    inc_step = tf.train.get_global_step().assign_add(batch_size)
    grads_and_vars = self._build_grads(worker_device=worker_device,
                                       device_setter=device_setter)
    self._train_op = tf.group(optimizer.apply_gradients(grads_and_vars),
                              inc_step)


  @tfu.scoped
  def build_summaries(self):
    self._summaries = self._build_summaries()

  @tfu.scoped
  def build_sync_ops(self):
    ops = self._build_sync_ops()
    if len(ops) > 0:
      self._sync_ops = tf.group(*ops)

  def start_training(self, sess, summary_writer, summary_period):
    if self.sync_ops is not None:
      sess.run(self.sync_ops)
    self._start_training(sess, summary_writer, summary_period)

  def get_feed_dict(self, sess):
    return self._get_feed_dict(sess)

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
      self.build_loss(worker_device, device_setter)
      self.build_train_op(optimizer, worker_device, device_setter)
      self.build_summaries()
      self.build_sync_ops()

  @abc.abstractmethod
  def _build_loss(self, worker_device=None, device_setter=None):
    ...

  @abc.abstractmethod
  def _build_grads(self, worker_device=None, device_setter=None):
    ...

  @abc.abstractmethod
  def _build_summaries(self):
    ...

  def _build_sync_ops(self):
    ops = []
    if self.local_policy is not None:
      ops = [
          v1.assign(v2) for v1, v2 in zip(self._local_policy.var_list(),
                                          self._global_policy.var_list())
      ]
    return ops

  @abc.abstractmethod
  def _start_training(self, sess, summary_writer, summary_period):
    ...

  @abc.abstractmethod
  def _get_feed_dict(self, sess):
    ...


class AlgorithmWrapper(BaseAlgorithm):
  def __init__(self, algorithm):
    super(AlgorithmWrapper, self).__init__(algorithm.global_policy,
                                           algorithm.local_policy)
    self.algorithm = algorithm
    assert self.scope is not None

  @property
  def batch_size(self):
    return self.algorithm.batch_size

  @property
  def logging_fetches(self):
    return self.algorithm.logging_fetches

  def _build_loss(self, worker_device=None, device_setter=None):
    self.algorithm.build_loss(worker_device=worker_device,
                              device_setter=device_setter)

  def _build_grads(self, worker_device=None, device_setter=None):
    self.algorithm.build_grads(worker_device=worker_device,
                               device_setter=device_setter)

  def _build_summaries(self):
    self.algorithm.build_summaries()

  def _build_sync_ops(self):
    self.algorithm.build_sync_ops()

  def _start_training(self, sess, summary_writer, summary_period):
    self.algorithm.start_training(sess, summary_writer, summary_period)

  def _get_feed_dict(self, sess):
    return self.algorithm.get_feed_dict(sess)
