from logging import getLogger

import numpy as np
import tensorflow as tf

from .base import BaseAlgorithm
from rl.utils.tf_utils import huber_loss


__all__ = ["DQNAlgorithm"]
logger = getLogger("rl")


def slice_with_actions(tensor, actions):
  """ Slices given tensor with actions along the second dimension. """
  tf.assert_integer(actions)
  batch_range = tf.range(tf.shape(tensor)[0])
  indices = tf.stack([batch_range, actions], axis=-1)
  return tf.gather_nd(tensor, indices)


class DQNAlgorithm(BaseAlgorithm):
  def __init__(self, policy, experience_replay, target_update_period,
               gamma=0.99, name=None):
    super(DQNAlgorithm, self).__init__(global_policy=policy,
                                       local_policy=None,
                                       name=name)
    self._experience_replay = experience_replay
    self._target_update_step = None
    self._target_update_period = target_update_period
    self._gamma = gamma

  @property
  def logging_fetches(self):
    return {"loss": self.loss}

  def _build_loss(self):
    self._actions_ph = tf.placeholder(tf.int32, [None], name="actions")
    self._rewards_ph = tf.placeholder(tf.float32, [None], name="rewards")
    self._resets_ph = tf.placeholder(tf.float32, [None], name="resets")

    with tf.variable_scope("loss"):
      with tf.variable_scope("predictions"):
        self._predictions = slice_with_actions(self.acting_policy.values,
                                               self._actions_ph)

      with tf.variable_scope("targets"):
        next_step_predictions = tf.reduce_max(
            self.acting_policy.target.values, axis=-1)
        self._next_step_predictions = (
            (1 - self._resets_ph) * self._gamma * next_step_predictions)
        self._targets = self._rewards_ph + self._next_step_predictions

        loss = tf.reduce_mean(huber_loss(self._targets - self._predictions))
        tf.assert_scalar(loss)
        return loss

  def _build_train_op(self, optimizer):
    self._loss_gradients = tf.gradients(
        self.loss, self.acting_policy.var_list())
    grads_and_vars = list(zip(
      self.global_policy.preprocess_gradients(self._loss_gradients),
      self.global_policy.var_list()
    ))
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op

  def _build_summaries(self):
    with tf.variable_scope("summaries") as scope:
      tf.summary.scalar("loss", self.loss)
      tf.summary.scalar("epsilon", self.acting_policy.epsilon)
      tf.summary.scalar("loss_gradient_norm",
                        tf.global_norm(self._loss_gradients))
      tf.summary.scalar("policy_norm",
                        tf.global_norm(self.acting_policy.var_list()))
      tf.summary.scalar("predicted_value",
                        tf.reduce_mean(self._predictions))
      tf.summary.scalar("target_value", tf.reduce_mean(self._targets))
      tf.summary.scalar("reward", tf.reduce_mean(self._rewards_ph))
      tf.summary.scalar("reset", tf.reduce_mean(self._resets_ph))
      tf.summary.scalar("next_value",
                        tf.reduce_mean(self._next_step_predictions))
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name)
      return tf.summary.merge(summaries)

  def build_target_update_ops(self):
    policy = self.acting_policy
    self._target_update_ops = tf.group(*[
        v1.assign(v2) for v1, v2 in zip(policy.target.var_list(),
                                        policy.var_list())
    ])

  def _build(self, *args, **kwargs):
    super(DQNAlgorithm, self)._build(*args, **kwargs)
    self.build_target_update_ops()

  def _get_feed_dict(self, sess, summaries=False):
    step = sess.run(tf.train.get_global_step())
    if step > 0 and self._target_update_step is None:
      # Training was restored.
      self._target_update_step = step - step % self._target_update_period
    if self._target_update_step is None or\
        step - self._target_update_step >= self._target_update_period:
      logger.info("Updating target policy on step #{}".format(step))
      sess.run(self._target_update_ops)
      self._target_update_step = step
    experience = self._experience_replay.next()
    policy = self.acting_policy
    feed_dict = {
        policy.observations: experience["observations"],
        self._actions_ph: experience["actions"],
        self._rewards_ph: experience["rewards"],
        self._resets_ph: experience["resets"].astype(np.float32),
        policy.target.observations: experience["next_observations"],
    }
    return feed_dict
