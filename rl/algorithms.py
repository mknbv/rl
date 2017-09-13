import abc
from contextlib import contextmanager
import logging
import numpy as np
import os
import queue
import tensorflow as tf

import rl.policies
from rl.trajectory import GAE, TrajectoryProducer

USE_DEFAULT = object()


class BaseAlgorithm(abc.ABC):
  @property
  @abc.abstractmethod
  def batch_size(self):
    ...

  @property
  @abc.abstractmethod
  def logging_fetches(self):
    ...

  @property
  @abc.abstractmethod
  def summaries(self):
    ...

  @property
  @abc.abstractmethod
  def grads_and_vars(self):
    ...

  @abc.abstractmethod
  def start_training(self, sess, summary_writer, summary_period):
    ...

  @abc.abstractmethod
  def get_feed_dict(self, sess):
    ...


class A3CAlgorithm(BaseAlgorithm):

  def __init__(self,
               env, *,
               trajectory_length,
               global_policy,
               local_policy=None,
               advantage_estimator=USE_DEFAULT,
               queue=None,
               entropy_coef=0.01,
               value_loss_coef=0.25,
               name=None):
    if local_policy is not None:
      self._sync_ops = tf.group(*[
          v1.assign(v2)
          for v1, v2 in zip(local_policy.var_list(), global_policy.var_list())
        ])
    else:
      self._sync_ops = None
    self._global_policy = global_policy
    self._local_policy = local_policy or global_policy
    self._trajectory_producer = TrajectoryProducer(
        env, self._local_policy, trajectory_length, queue)
    if advantage_estimator == USE_DEFAULT:
      advantage_estimator = GAE(policy=self._local_policy)
    self._advantage_estimator = advantage_estimator
    if name is None:
      name = self.__class__.__name__
    with tf.variable_scope(None, name) as scope:
      self._scope = scope
      self._global_step = tf.train.get_global_step()
      self._actions = tf.placeholder(self._global_policy.distribution.dtype,
                                    [None], name="actions")
      self._advantages = tf.placeholder(tf.float32, [None], name="advantages")
      self._value_targets = tf.placeholder(tf.float32, [None],
                                           name="value_targets")
      with tf.name_scope("loss"):
        pd = self._local_policy.distribution
        with tf.name_scope("policy_loss"):
          self._policy_loss = tf.reduce_sum(
              pd.neglogp(self._actions) * self._advantages)
          self._policy_loss -= entropy_coef * tf.reduce_sum(pd.entropy())
        with tf.name_scope("value_loss"):
          self._v_loss = tf.reduce_sum(tf.square(
              tf.squeeze(self._local_policy.value_preds) - self._value_targets))
        self._loss = self._policy_loss + value_loss_coef * self._v_loss
        self._gradients = tf.gradients(
            self._loss, self._local_policy.var_list())
        self._grads_and_vars = zip(
            self._global_policy.preprocess_gradients(self._gradients),
            self._global_policy.var_list()
          )
      self._init_summaries()

  def _init_summaries(self):
    with tf.variable_scope("summaries") as scope:
      tf.summary.scalar("value_preds",
                        tf.reduce_mean(self._local_policy.value_preds))
      tf.summary.scalar("value_targets",
                        tf.reduce_mean(self._value_targets))
      tf.summary.scalar(
          "distribution_entropy",
          tf.reduce_mean(self._local_policy.distribution.entropy()))
      batch_size = tf.to_float(tf.shape(self._actions)[0])
      tf.summary.scalar("policy_loss", self._policy_loss / batch_size)
      tf.summary.scalar("value_loss", self._v_loss / batch_size)
      tf.summary.scalar("loss", self._loss / batch_size)
      tf.summary.scalar("gradient_norm", tf.global_norm(self._gradients))
      tf.summary.scalar(
          "policy_norm", tf.global_norm(self._local_policy.var_list()))
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name)
      self._summaries = tf.summary.merge(summaries)

  @property
  def batch_size(self):
    return tf.shape(self._local_policy.inputs)[0]

  @property
  def logging_fetches(self):
    return {
        "Policy loss": self._policy_loss,
        "Value loss": self._v_loss
      }

  @property
  def summaries(self):
    return self._summaries

  @property
  def grads_and_vars(self):
    return self._grads_and_vars

  def start_training(self, sess, summary_writer, summary_period):
    if self._sync_ops is not None:
      sess.run(self._sync_ops)
    self._trajectory_producer.start(summary_writer, summary_period, sess=sess)

  def get_feed_dict(self, sess):
    if self._sync_ops is not None:
      sess.run(self._sync_ops)
    trajectory = self._trajectory_producer.next()
    advantages, value_targets = self._advantage_estimator(trajectory, sess=sess)
    feed_dict = {
        self._local_policy.inputs:
            trajectory.observations[:trajectory.num_timesteps],
        self._actions: trajectory.actions[:trajectory.num_timesteps],
        self._advantages: advantages,
        self._value_targets: value_targets
    }
    if self._local_policy.state is not None:
      feed_dict[self._local_policy.state_in] = trajectory.policy_states[0]
    return feed_dict
