import numpy as np
import tensorflow as tf

from rl.algorithms import BaseAlgorithm
from rl.algorithms.advantages import GAE
from rl.utils.tf_utils import explained_variance

USE_DEFAULT = object()


class PPO2Algorithm(BaseAlgorithm):
  def __init__(self,
               interactions_producer,
               global_policy,
               local_policy=None,
               num_epochs=4,
               num_minibatches=4,
               advantage_estimator=USE_DEFAULT,
               cliprange=0.2,
               value_loss_coef=0.25,
               entropy_coef=0.01,
               max_grad_norm=0.5,
               name=None):
    super(PPO2Algorithm, self).__init__(global_policy=global_policy,
                                        local_policy=local_policy, name=name)
    self._interactions_producer = interactions_producer
    self._num_epochs = num_epochs
    self._epoch_count = 0
    self._num_minibatches = num_minibatches
    self._minibatch_count = 0
    self._trajectory = None
    if advantage_estimator == USE_DEFAULT:
      advantage_estimator = GAE(self.acting_policy, normalize=True)
    self._advantage_estimator = advantage_estimator
    self._cliprange = cliprange
    self._value_loss_coef = value_loss_coef
    self._entropy_coef = entropy_coef
    self._max_grad_norm = max_grad_norm

  @property
  def logging_fetches(self):
    return {
        "policy_loss": self._policy_loss,
        "value_loss": self._value_loss,
        "loss": self.loss
    }

  def _build_loss(self):
    action_space = self._interactions_producer.action_space
    act_type = action_space.dtype
    act_shape = (None,) + action_space.shape
    self._actions_ph = tf.placeholder(act_type, act_shape, name="actions")
    self._advantages_ph = tf.placeholder(tf.float32, [None],
                                         name="advantages")
    self._value_targets_ph = tf.placeholder(tf.float32, [None, 1],
                                            name="value_targets")
    self._old_log_prob_ph = tf.placeholder(tf.float32, [None],
                                           name="old_log_prob")
    self._old_value_preds_ph = tf.placeholder(tf.float32, [None, 1],
                                              name="old_value_preds")

    with tf.variable_scope("policy"):
      policy = self.acting_policy
      logp = policy.distribution.log_prob(self._actions_ph)

      self._ratio = tf.exp(logp - self._old_log_prob_ph)
      self._policy_loss_regular = -self._ratio * self._advantages_ph
      self._policy_loss_clipped = -(
          tf.clip_by_value(self._ratio,
                           1. - self._cliprange,
                           1. + self._cliprange)
          * self._advantages_ph
      )
      self._policy_loss = (
          tf.reduce_mean(tf.maximum(self._policy_loss_regular,
                                    self._policy_loss_clipped))
          - self._entropy_coef * tf.reduce_mean(policy.distribution.entropy())
      )

      self._approxkl = 0.5 * tf.reduce_mean(
          tf.square(logp - self._old_log_prob_ph))
      self._clipfrac = tf.reduce_mean(
          tf.to_float(tf.abs(self._ratio - 1) > self._cliprange))

    with tf.variable_scope("value"):
      value_preds_clipped = (
          self._old_value_preds_ph
          + tf.clip_by_value(policy.critic_tensor - self._old_value_preds_ph,
                             -self._cliprange, self._cliprange)
      )
      self._value_loss_regular = tf.square(policy.critic_tensor
                                           - self._value_targets_ph)
      self._value_loss_clipped = tf.square(value_preds_clipped
                                           - self._value_targets_ph)
      self._value_loss = tf.reduce_mean(tf.maximum(self._value_loss_regular,
                                                   self._value_loss_clipped))

    return self._policy_loss + self._value_loss_coef * self._value_loss

  def _build_train_op(self, optimizer):
    self._loss_gradients = tf.gradients(self.loss,
                                        self.acting_policy.var_list())
    if self._max_grad_norm is not None:
      gradients, _ = tf.clip_by_global_norm(self._loss_gradients,
                                            self._max_grad_norm)
    else:
      gradients = self.global_policy.preprocess_gradients(self._loss_gradients)
    grads_and_vars = list(zip(gradients, self.global_policy.var_list()))
    return optimizer.apply_gradients(grads_and_vars)

  def _build_summaries(self):
    with tf.variable_scope("summaries"):
      policy = self.acting_policy
      tf.summary.scalar("policy_entropy",
                        tf.reduce_mean(policy.distribution.entropy()))
      tf.summary.scalar("policy_norm", tf.global_norm(policy.var_list()))
      tf.summary.histogram("ratio", self._ratio)
      tf.summary.scalar("policy_loss", self._policy_loss)
      tf.summary.scalar("advantages", tf.reduce_mean(self._advantages_ph))
      tf.summary.histogram("advantages", self._advantages_ph)
      tf.summary.scalar("value_loss", self._value_loss)
      tf.summary.scalar("loss", tf.reduce_mean(self.loss))
      tf.summary.scalar("loss_grad_norm", tf.global_norm(self._loss_gradients))
      tf.summary.scalar("clip_fraction", self._clipfrac)
      tf.summary.scalar("approx_KL", self._approxkl)
      tf.summary.scalar(
          "critic_explained_variance",
          explained_variance(policy.critic_tensor, self._value_targets_ph)
      )
      tf.summary.histogram("value_preds_hist", policy.critic_tensor)
      tf.summary.histogram("value_targets_hist", self._value_targets_ph)
      tf.summary.scalar("value_preds", tf.reduce_mean(policy.critic_tensor))
      tf.summary.scalar("value_targets",
                        tf.reduce_mean(self._value_targets_ph))
      name_scope = tf.get_default_graph().get_name_scope()
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=name_scope)
      return tf.summary.merge(summaries)

  def _pick_state_inputs(self, indices):
    state = self._trajectory["state"]
    if self.acting_policy.state_inputs not in state:
      raise ValueError("Trajectory does not contain policy state inputs")
    state_inputs = state[self.acting_policy.state_inputs]
    if isinstance(state_inputs, np.ndarray):
      return state_inputs[indices]
    elif isinstance(state_inputs, tf.nn.rnn_cell.LSTMStateTuple):
      return tf.nn.rnn_cell.LSTMStateTuple(c=state_inputs.c[indices],
                                           h=state_inputs.h[indices])
    else:
      raise TypeError("Trajectory contains unsupported policy "
                      "state_inputs type: {}".format(type(state_inputs)))

  def _shuffle_trajectory(self):
    nenvs = self.acting_policy.state_inputs.shape[0].value
    env_steps = self._trajectory["state"]["env_steps"]
    if self.acting_policy.state_inputs is None:
      indices = np.random.permutation(env_steps)
    else:
      env_indices = np.random.permutation(nenvs)
      self._trajectory["state"][self.acting_policy.state_inputs] =\
          self._pick_state_inputs(env_indices)
      indices = np.ravel(env_indices + np.arange(0, env_steps, nenvs)[:, None])

    for key, val in filter(lambda kv: kv[0] != "state",
                           self._trajectory.items()):
      self._trajectory[key] = val[indices]

  def _get_feed_dict(self, sess, summaries=False):
    sess.run(self.sync_op)

    epoch_update = self._minibatch_count == self._num_minibatches
    if epoch_update:
      self._epoch_count += 1
      self._minibatch_count = 0

    if self._trajectory is None or self._epoch_count == self._num_epochs:
      self._trajectory = self._interactions_producer.next()
      advantages, value_targets = self._advantage_estimator(self._trajectory,
                                                            sess)
      self._trajectory["advantages"] = advantages
      self._trajectory["value_targets"] = value_targets
      self._epoch_count = 0

    if epoch_update or self._epoch_count == 0:
      self._shuffle_trajectory()

    env_steps = self._trajectory["state"]["env_steps"]
    policy = self.acting_policy
    if policy.state_inputs is None:
      minibatch_size = env_steps // self._num_minibatches
      minibatch_start = self._minibatch_count * minibatch_size
      minibatch_indices = np.arange(minibatch_start,
                                    min(minibatch_start + minibatch_size,
                                        env_steps))
    else:
      if self._interactions_producer.num_envs % self._num_minibatches != 0:
        raise ValueError(
            "num_minibatches = {} does not divide "
            "interaction_producer.num_envs = {}"
            .format(num_minibatches, interactions_producer.num_envs))
      nenvs = policy.state_inputs.shape[0].value
      envs_per_sample = nenvs // self._num_minibatches
      env_indices_start = self._minibatch_count * envs_per_sample
      env_indices = np.arange(env_indices_start,
                              env_indices_start + envs_per_sample)
      minibatch_indices = np.ravel(env_indices
                                   + np.arange(0, env_steps, nenvs)[:, None])

    self._minibatch_count += 1
    feed_dict = {
      self._actions_ph: self._trajectory["actions"][minibatch_indices],
      self._value_targets_ph:
        self._trajectory["value_targets"][minibatch_indices],  # noqa: E131
      self._advantages_ph: self._trajectory["advantages"][minibatch_indices],
      self._old_log_prob_ph: self._trajectory["log_probs"][minibatch_indices],
      self._old_value_preds_ph:
        self._trajectory["critic_values"][minibatch_indices],
    }
    if policy.state_inputs is not None:
      state_inputs = self._pick_state_inputs(env_indices)
    else:
      state_inputs = None
    feed_dict.update(policy.get_feed_dict(
        self._trajectory["observations"][minibatch_indices],
        state_inputs=state_inputs
    ))

    return feed_dict
