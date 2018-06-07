import tensorflow as tf

from .base import BaseAlgorithm
from .advantages import ActorCriticAdvantage
from rl.utils.tf_utils import explained_variance


__all__ = ["A3CAlgorithm"]

USE_DEFAULT = object()


class A3CAlgorithm(BaseAlgorithm):

  def __init__(self,
               interactions_producer,
               global_policy,
               local_policy=None,
               advantage_estimator=USE_DEFAULT,
               entropy_coef=0.01,
               value_loss_coef=0.25,
               max_grad_norm=40,
               name=None):
    super(A3CAlgorithm, self).__init__(global_policy, local_policy, name=name)
    self._interactions_producer = interactions_producer
    if advantage_estimator == USE_DEFAULT:
      advantage_estimator = ActorCriticAdvantage(
          policy=self.local_or_global_policy)
    self._advantage_estimator = advantage_estimator
    self._entropy_coef = entropy_coef
    self._value_loss_coef = value_loss_coef
    self._max_grad_norm = max_grad_norm

  @property
  def logging_fetches(self):
    return {
        "Policy loss": self._policy_loss,
        "Value loss": self._value_loss
      }

  def _build_loss(self, worker_device=None, device_setter=None):
    policy = self.local_or_global_policy
    act_type = self._interactions_producer.action_space.dtype
    act_shape = (None,) + self._interactions_producer.action_space.shape
    self._actions = tf.placeholder(act_type, act_shape, name="actions")
    self._advantages = tf.placeholder(tf.float32, [None], name="advantages")
    self._value_targets = tf.placeholder(tf.float32,
                                         policy.critic_tensor.shape,
                                         name="value_targets")
    with tf.name_scope("loss"):
      with tf.name_scope("policy_loss"):
        self._policy_loss = -tf.reduce_sum(
            policy.distribution.log_prob(self._actions) * self._advantages
            + self._entropy_coef * policy.distribution.entropy()
        )
      with tf.name_scope("value_loss"):
        self._value_loss = tf.reduce_sum(tf.square(
            self.local_or_global_policy.critic_tensor
            - self._value_targets
        ))
      loss = self._policy_loss + self._value_loss_coef * self._value_loss
      return loss

  def _build_train_op(self, optimizer):
    self._loss_gradients =\
        tf.gradients(self._loss, self.local_or_global_policy.var_list())
    if self._max_grad_norm is None:
      preprocessed_gradients = self._global_policy.preprocess_gradients(
          self._loss_gradients)
    else:
      preprocessed_gradients, _ = tf.clip_by_global_norm(
          self._loss_gradients, self._max_grad_norm)
    grads_and_vars = list(zip(
        preprocessed_gradients,
        self._global_policy.var_list()
    ))
    with tf.device(self.global_policy.var_list()[0].device):
      train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op

  def _build_summaries(self):
    with tf.variable_scope("summaries") as scope:
      policy = self.local_or_global_policy
      tf.summary.scalar(
          "critic_explained_variance",
          explained_variance(
            self._value_targets[:, 0],
            policy.critic_tensor[:, 0]
          )
      )
      tf.summary.scalar("critic_values", tf.reduce_mean(policy.critic_tensor))
      tf.summary.scalar("value_targets", tf.reduce_mean(self._value_targets))
      tf.summary.scalar("advantages", tf.reduce_mean(self._advantages))
      tf.summary.scalar("distribution_entropy",
                        tf.reduce_mean(policy.distribution.entropy()))
      tf.summary.scalar("policy_loss", self._policy_loss)
      tf.summary.scalar("value_loss", self._value_loss)
      tf.summary.scalar("loss", self._loss)
      tf.summary.scalar("loss_gradient_norm",
                        tf.global_norm(self._loss_gradients))
      tf.summary.scalar("policy_norm",
                        tf.global_norm(policy.var_list()))
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name)
      return tf.summary.merge(summaries)

  def _get_feed_dict(self, sess, summary_time=False):
    sess.run(self.sync_ops)
    trajectory = self._interactions_producer.next()
    advantages, value_targets = self._advantage_estimator(trajectory,
                                                          sess=sess)
    feed_dict = {
        self.local_or_global_policy.observations: trajectory["observations"],
        self._actions: trajectory["actions"],
        self._advantages: advantages,
        self._value_targets: value_targets
    }
    policy = self.local_or_global_policy
    if policy.state_inputs is not None:
      feed_dict[policy.state_inputs] = trajectory["state"][policy.state_inputs]
    return feed_dict
