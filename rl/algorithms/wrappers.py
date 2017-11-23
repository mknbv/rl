import tensorflow as tf

from rl.algorithms import BaseAlgorithm
import rl.utils.tf_utils as tfu


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
