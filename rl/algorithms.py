import abc
from contextlib import contextmanager
import logging
import numpy as np
import os
import tensorflow as tf

import rl.policies
from rl.trajectory import GAE, TrajectoryProducer
import rl.tf_utils as tfu

USE_DEFAULT = object()


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
    self._grads_and_vars = None
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
  def batch_size(self):
    return tf.shape(self.local_or_global_policy.inputs)[0]

  @property
  @abc.abstractmethod
  def logging_fetches(self):
    ...

  @property
  def loss(self):
    return self._loss

  @property
  def grads_and_vars(self):
    return self._grads_and_vars

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
  def build_grads(self, worker_device=None, device_setter=None):
    self._grads_and_vars = self._build_grads(worker_device=worker_device,
                                             device_setter=device_setter)

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

  def _build(self, worker_device=None, device_setter=None):
    """ Adds all the variables and ops needed by this algorithm.

    This currently does not support partially build algorithms, so
    if, for example, `build_loss` was called before calling this
    function, it might fail in weird ways.  `worker_device` and
    `device_setter` are required when algorithm implements
    `local_policy` (`algorithm.local_policy is not None`). When
    `local_policy` is not implemented only worker_device is used.

    Args:
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
      self.build_grads(worker_device, device_setter)
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


class A3CAlgorithm(BaseAlgorithm):

  def __init__(self,
               trajectory_producer,
               global_policy,
               local_policy=None,
               advantage_estimator=USE_DEFAULT,
               entropy_coef=0.01,
               value_loss_coef=0.25,
               name=None):
    super(A3CAlgorithm, self).__init__(global_policy, local_policy, name=name)
    self._trajectory_producer = trajectory_producer
    if advantage_estimator == USE_DEFAULT:
      advantage_estimator = GAE(policy=self.local_or_global_policy)
    self._advantage_estimator = advantage_estimator
    self._entropy_coef = entropy_coef
    self._value_loss_coef = value_loss_coef

  @property
  def logging_fetches(self):
    return {
        "Policy loss": self._policy_loss,
        "Value loss": self._v_loss
      }

  def _build_loss(self, worker_device=None, device_setter=None):
    self._actions = tf.placeholder(self._global_policy.distribution.dtype,
                                   [None], name="actions")
    self._advantages = tf.placeholder(tf.float32, [None], name="advantages")
    self._value_targets = tf.placeholder(tf.float32, [None],
                                         name="value_targets")
    with tf.name_scope("loss"):
      pd = self.local_or_global_policy.distribution
      with tf.name_scope("policy_loss"):
        self._policy_loss = tf.reduce_sum(
            pd.neglogp(self._actions) * self._advantages)
        self._policy_loss -= self._entropy_coef\
            * tf.reduce_sum(pd.entropy())
      with tf.name_scope("value_loss"):
        self._v_loss = tf.reduce_sum(
            tf.square(
              tf.squeeze(self.local_or_global_policy.value_preds)
              - self._value_targets
            )
        )
      loss = self._policy_loss + self._value_loss_coef * self._v_loss
      return loss

  def _build_grads(self, worker_device=None, device_setter=None):
    with tf.device(worker_device):
      self._loss_gradients =\
          tf.gradients(self._loss, self.local_or_global_policy.var_list())
      grads_and_vars = list(zip(
          self._global_policy.preprocess_gradients(self._loss_gradients),
          self._global_policy.var_list()
      ))
      return grads_and_vars

  def _build_summaries(self):
    with tf.variable_scope("summaries") as scope:
      s = tf.summary.scalar(
          "value_preds",
          tf.reduce_mean(self.local_or_global_policy.value_preds))
      tf.summary.scalar("value_targets",
                        tf.reduce_mean(self._value_targets))
      tf.summary.scalar(
          "distribution_entropy",
          tf.reduce_mean(self.local_or_global_policy.distribution.entropy()))
      batch_size = tf.to_float(tf.shape(self._actions)[0])
      tf.summary.scalar("policy_loss", self._policy_loss / batch_size)
      tf.summary.scalar("value_loss", self._v_loss / batch_size)
      tf.summary.scalar("loss", self._loss / batch_size)
      tf.summary.scalar("loss_gradient_norm",
                        tf.global_norm(self._loss_gradients))
      tf.summary.scalar(
          "policy_norm", tf.global_norm(self.local_or_global_policy.var_list()))
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name)
      return tf.summary.merge(summaries)

  def _start_training(self, sess, summary_writer, summary_period):
    self._trajectory_producer.start(summary_writer, summary_period, sess=sess)

  def _get_feed_dict(self, sess):
    if self.sync_ops is not None:
      sess.run(self.sync_ops)
    trajectory = self._trajectory_producer.next()
    advantages, value_targets = self._advantage_estimator(trajectory, sess=sess)
    feed_dict = {
        self.local_or_global_policy.inputs:
            trajectory.observations[:trajectory.num_timesteps],
        self._actions: trajectory.actions[:trajectory.num_timesteps],
        self._advantages: advantages,
        self._value_targets: value_targets
    }
    if self.local_or_global_policy.state is not None:
      feed_dict[self.local_or_global_policy.state_in] =\
          trajectory.policy_states[0]
    return feed_dict
