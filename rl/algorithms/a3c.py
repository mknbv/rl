import tensorflow as tf
from gym import spaces

from .base import BaseAlgorithm
from .advantages import GAE


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
               sparse_rewards=True,
               name=None):
    super(A3CAlgorithm, self).__init__(global_policy, local_policy, name=name)
    self._interactions_producer = interactions_producer
    if advantage_estimator == USE_DEFAULT:
      advantage_estimator = GAE(policy=self.local_or_global_policy)
    self._advantage_estimator = advantage_estimator
    self._entropy_coef = entropy_coef
    self._value_loss_coef = value_loss_coef
    self._sparse_rewards = sparse_rewards

  @property
  def logging_fetches(self):
    return {
        "Policy loss": self._policy_loss,
        "Value loss": self._v_loss
      }

  def _build_loss(self, worker_device=None, device_setter=None):
    pd = self.local_or_global_policy.distribution
    act_shape = (None,)
    if not isinstance(self._interactions_producer.action_space,
                      spaces.Discrete):
      act_shape = act_shape + self._interactions_producer.action_space.shape
    self._actions = tf.placeholder(pd.dtype, act_shape, name="actions")
    self._advantages = tf.placeholder(tf.float32, [None], name="advantages")
    self._value_targets = tf.placeholder(tf.float32, [None],
                                         name="value_targets")
    with tf.name_scope("loss"):
      with tf.name_scope("policy_loss"):
        self._policy_loss = tf.reduce_sum(
            -pd.log_prob(self._actions) * self._advantages)
        self._policy_loss -= self._entropy_coef * tf.reduce_sum(pd.entropy())
      with tf.name_scope("value_loss"):
        if self._sparse_rewards:
          reducer = tf.reduce_sum
        else:
          reducer = tf.reduce_mean
        self._v_loss = reducer(
            tf.square(
              tf.squeeze(self.local_or_global_policy.critic_tensor)
              - self._value_targets
            )
        )
      loss = self._policy_loss + self._value_loss_coef * self._v_loss
      return loss

  def _build_train_op(self, optimizer):
    self._loss_gradients =\
        tf.gradients(self._loss, self.local_or_global_policy.var_list())
    grads_and_vars = list(zip(
        self._global_policy.preprocess_gradients(self._loss_gradients),
        self._global_policy.var_list()
    ))
    train_op = optimizer.apply_gradients(grads_and_vars)
    batch_size = tf.to_int64(
        tf.shape(self.local_or_global_policy.observations)[0])
    inc_step = tf.train.get_global_step().assign_add(batch_size)
    train_op = tf.group(train_op, inc_step)
    return train_op

  def _build_summaries(self):
    with tf.variable_scope("summaries") as scope:
      tf.summary.scalar(
          "critic_values",
          tf.reduce_mean(self.local_or_global_policy.critic_tensor))
      tf.summary.scalar("value_targets",
                        tf.reduce_mean(self._value_targets))
      tf.summary.scalar("advantages", tf.reduce_mean(self._advantages))
      tf.summary.scalar(
          "distribution_entropy",
          tf.reduce_mean(self.local_or_global_policy.distribution.entropy()))
      tf.summary.scalar("policy_loss", self._policy_loss)
      tf.summary.scalar("value_loss", self._v_loss)
      tf.summary.scalar("loss", self._loss)
      tf.summary.scalar("loss_gradient_norm",
                        tf.global_norm(self._loss_gradients))
      tf.summary.scalar(
          "policy_norm", tf.global_norm(self.local_or_global_policy.var_list()))
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name)
      return tf.summary.merge(summaries)

  def _start_training(self, sess, summary_writer, summary_period):
    self._interactions_producer.start(summary_writer, summary_period, sess=sess)

  def _get_feed_dict(self, sess, summary_time=False):
    if self.sync_ops is not None:
      sess.run(self.sync_ops)
    trajectory = self._interactions_producer.next()
    advantages, value_targets = self._advantage_estimator(trajectory, sess=sess)
    num_timesteps = trajectory["num_timesteps"]
    feed_dict = {
        self.local_or_global_policy.observations:
            trajectory["observations"][:num_timesteps],
        self._actions: trajectory["actions"][:num_timesteps],
        self._advantages: advantages,
        self._value_targets: value_targets
    }
    if self.local_or_global_policy.state_inputs is not None:
      feed_dict[self.local_or_global_policy.state_inputs] =\
          trajectory["policy_state"]
    return feed_dict
