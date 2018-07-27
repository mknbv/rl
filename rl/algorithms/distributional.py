import numpy as np
import tensorflow as tf

from rl.algorithms.dqn import DQNAlgorithm, slice_with_actions
from rl.utils.tf_utils import huber_loss


class DistributionalAlgorithm(DQNAlgorithm):
  def __init__(self, experience_replay, policy, target_update_period,
               kind="qr-dqn-1", gamma=0.99, name=None):
    super(DistributionalAlgorithm, self).__init__(
        policy, experience_replay,
        target_update_period=target_update_period, name=name)
    if kind not in ["qr-dqn-0", "qr-dqn-1"]:
      raise TypeError("kind must be one of ['qr-dqn-0', 'qr-dqn-1'], but is {}"
                      .format(kind))
    self._kind = kind

  def _build_loss(self):
    policy = self.local_or_global_policy
    self._actions_ph = tf.placeholder(tf.int32, [None], name="actions")
    self._rewards_ph = tf.placeholder(tf.float32, [None], name="rewards")
    self._resets_ph = tf.placeholder(tf.float32, [None], name="resets")

    with tf.variable_scope("loss"):
      with tf.variable_scope("predictions"):
        self._predictions = slice_with_actions(policy.output_tensor,
                                               self._actions_ph)

      with tf.variable_scope("targets"):
        self._next_actions = tf.cast(
            tf.argmax(policy.target.values, axis=-1), tf.int32)
        all_next_step_predictions = slice_with_actions(
            policy.target.output_tensor, self._next_actions)
        next_step_multiplier = (1 - self._resets_ph)[...,None]
        self._next_step_predictions = (
            next_step_multiplier * self._gamma * all_next_step_predictions)
        self._targets = self._rewards_ph[...,None] + self._next_step_predictions

      with tf.variable_scope("quantile_loss"):
        nbins = policy.output_tensor.shape[-1].value
        cdf = np.arange(0, nbins+1) / nbins
        midpoints = (cdf[:-1] + cdf[1:]) / 2
        overestimation = tf.to_float(
            self._targets[...,None] < self._predictions[:,None])

        if self._kind == "qr-dqn-0":
          loss = tf.reduce_sum(
              tf.reduce_mean(
                (self._targets[...,None] - self._predictions[:,None])
                * (midpoints[None,None] - overestimation),
                axis=[0,1]
              ),
              axis=0
          )
        else:
          assert self._kind == "qr-dqn-1", self._kind
          loss = tf.reduce_sum(
              tf.reduce_mean(
                huber_loss(self._targets[...,None] - self._predictions[:,None])
                * tf.abs(midpoints[None,None] - overestimation),
                axis=[0,1]
              ),
              axis=0
          )
        tf.assert_scalar(loss)
        return loss
