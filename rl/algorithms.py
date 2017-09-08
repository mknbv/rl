from contextlib import contextmanager
import logging
import numpy as np
import os
import queue
import tensorflow as tf

import rl.policies
from rl.trajectory import GAE, TrajectoryProducer

USE_DEFAULT = object()


class BaseA3CAlgorithm(object):

  def __init__(self,
               env, *,
               trajectory_length,
               global_policy,
               local_policy=None,
               advantage_estimator=USE_DEFAULT,
               queue=queue.Queue(maxsize=5),
               entropy_coef=0.01,
               value_loss_coef=0.25,
               name=None):
    if local_policy is not None:
      self.sync_ops = tf.group(*[
          v1.assign(v2)
          for v1, v2 in zip(local_policy.var_list(), global_policy.var_list())
        ])
    else:
      self.sync_ops = None
    self.global_policy = global_policy
    self.local_policy = local_policy or global_policy
    self.trajectory_producer = TrajectoryProducer(
        env, self.local_policy, trajectory_length, queue)
    if advantage_estimator == USE_DEFAULT:
      advantage_estimator = GAE(policy=self.local_policy)
    self.advantage_estimator = advantage_estimator
    if name is None:
      name = self.__class__.__name__
    with tf.variable_scope(None, name) as scope:
      self.scope = scope
      self.global_step = tf.train.get_global_step()
      self.actions = tf.placeholder(self.global_policy.distribution.dtype,
                                    [None], name="actions")
      self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
      self.value_targets = tf.placeholder(tf.float32, [None],
                                          name="value_targets")
      with tf.name_scope("loss"):
        pd = self.local_policy.distribution
        with tf.name_scope("policy_loss"):
          self.policy_loss = tf.reduce_sum(
              pd.neglogp(self.actions) * self.advantages)
          self.policy_loss -= entropy_coef * tf.reduce_sum(pd.entropy())
        with tf.name_scope("value_loss"):
          self.v_loss = tf.reduce_sum(tf.square(
              tf.squeeze(self.local_policy.value_preds) - self.value_targets))
        self.loss = self.policy_loss + value_loss_coef * self.v_loss
        self.gradients = tf.gradients(self.loss, self.local_policy.var_list())
        self._grads_and_vars = zip(
            self.global_policy.preprocess_gradients(self.gradients),
            self.global_policy.var_list()
          )
      self._init_summaries()

  def _init_summaries(self):
    with tf.variable_scope("summaries") as scope:
      tf.summary.scalar("value_preds",
                        tf.reduce_mean(self.local_policy.value_preds))
      tf.summary.scalar("value_targets",
                        tf.reduce_mean(self.value_targets))
      tf.summary.scalar(
          "distribution_entropy",
          tf.reduce_mean(self.local_policy.distribution.entropy()))
      batch_size = tf.to_float(tf.shape(self.actions)[0])
      tf.summary.scalar("policy_loss", self.policy_loss / batch_size)
      tf.summary.scalar("value_loss", self.v_loss / batch_size)
      tf.summary.scalar("loss", self.loss / batch_size)
      tf.summary.scalar("gradient_norm", tf.global_norm(self.gradients))
      tf.summary.scalar(
          "policy_norm", tf.global_norm(self.local_policy.var_list()))
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name)
      self.summaries = tf.summary.merge(summaries)

  @property
  def batch_size(self):
    return tf.shape(self.local_policy.inputs)[0]

  @property
  def logging_fetches(self):
    return {
        "Policy loss": self.policy_loss,
        "Value loss": self.v_loss
      }

  @property
  def grads_and_vars(self):
    return self._grads_and_vars

  def start_training(self, sess, summary_writer, summary_period):
    if self.sync_ops is not None:
      sess.run(self.sync_ops)
    self.trajectory_producer.start(summary_writer, summary_period, sess=sess)

  def get_feed_dict(self, sess):
    if self.sync_ops is not None:
      sess.run(self.sync_ops)
    trajectory = self.trajectory_producer.next()
    advantages, value_targets = self.advantage_estimator(trajectory, sess=sess)
    feed_dict = {
        self.local_policy.inputs:
            trajectory.observations[:trajectory.num_timesteps],
        self.actions: trajectory.actions[:trajectory.num_timesteps],
        self.advantages: advantages,
        self.value_targets: value_targets
    }
    if self.local_policy.get_state() is not None:
      feed_dict[self.local_policy.state_in] = trajectory.policy_states[0]
    return feed_dict
