import tensorflow as tf

from rl.algorithms import BaseAlgorithm
import rl.policies
from rl.trajectory import GAE, TrajectoryProducer
import rl.utils.tf_utils as tfu

USE_DEFAULT = object()


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
            -pd.log_prob(self._actions) * self._advantages)
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
      tf.summary.scalar("advantages", tf.reduce_mean(self._advantages))
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
