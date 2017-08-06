from collections import namedtuple
from datetime import datetime
import logging
import numpy as np
import os
import tensorflow as tf

import rl.policies


class Trajectory(object):
  def __init__(self, env, policy, num_timesteps):
    self.env = env
    self.policy = policy
    self.latest_observation = self.env.reset()
    self.num_timesteps = num_timesteps
    self.done = False
    obs_shape = list(self.latest_observation.shape)
    obs_type = self.latest_observation.dtype
    self.observations = np.zeros([num_timesteps] + obs_shape, dtype=obs_type)
    self.resets = np.zeros([num_timesteps], dtype=np.bool)
    self.rewards = np.zeros([num_timesteps])
    act_shape = list(self.policy.distribution.shape[1:])
    act_type = self.policy.distribution.dtype.as_numpy_dtype
    self.actions = np.zeros([num_timesteps] + act_shape, dtype=act_type)
    self.episode_count = 0
    self.policy_states = None
    if self.policy.get_state() is not None:
      self.policy_states = [None] * self.num_timesteps
    if isinstance(self.policy, rl.policies.CNNPolicy):
      self.summaries = tf.summary.image("Trajectory/observation",
                                        self.policy.inputs)
    else:
      self.summaries = None
    self._init_stats()

  def _init_stats(self):
    pass

  def _update_policy_stats(self, index, observation, sess=None):
    self.actions[index] = self.policy.act(observation, sess=sess)

  def _add_summary(self, info, summary_writer, sess=None):
    if sess is None:
      sess = tf.get_default_session()
    summary = tf.Summary()
    with tf.variable_scope(None, self.__class__.__name__):
      logkeys = filter(lambda k: k.startswith("logging"), info.keys())
      for key in logkeys:
        val = float(info[key])
        tag = "Trajectory/" + key.split(".", 1)[1]
        summary.value.add(tag=tag, simple_value=val)
    logging.info("Episode #{} finished, reward: {}"\
        .format(self.episode_count, info["logging.total_reward"]))
    step = sess.run(tf.train.get_global_step())
    summary_writer.add_summary(summary, step)
    if self.summaries is not None:
      fetched_summary = sess.run(self.summaries,
                                 {self.policy.inputs: self.observations})
      summary_writer.add_summary(fetched_summary, step)

  def update(self, summary_writer, sess=None):
    num_timesteps = self.num_timesteps
    for i in range(num_timesteps):
      self.observations[i] = self.latest_observation
      if self.policy_states is not None:
        self.policy_states[i] = self.policy.get_state()
      self._update_stats(i, self.latest_observation, sess)
      self.latest_observation, self.rewards[i], self.resets[i], info =\
          self.env.step(self.actions[i])
      if self.resets[i]:
        self.latest_observation = self.env.reset()
        self.policy.reset()
        self._add_summary(info, summary_writer, sess)
        self.episode_count += 1


class ValueFunctionTrajectory(Trajectory):
  def __init__(self, env, policy, num_timesteps):
    if not isinstance(policy, rl.policies.ValueFunctionPolicy):
      raise ValueError("policy must iherit from ValueFunctionPolicy")
    super(ValueFunctionTrajectory, self)\
        .__init__(env, policy, num_timesteps)

  def _init_stats(self):
    self.value_preds = np.zeros([self.num_timesteps], dtype=np.float32)

  def _update_stats(self, index, observation, sess=None):
    self.actions[index], self.value_preds[index] =\
        self.policy.act(observation, sess=sess)


# def value_function_critic(policy, trajectory, gamma=0.99, sess=None):
#   if not isinstance(policy, rl.policies.ValueFunctionPolicy):
#     raise TypeError("policy must be an instance of ValueFunctionPolicy")
#   if not isinstance(trajectory, ValueFunctionTrajectory):
#     raise TypeError("trajectory must be an instance of ValueFunctionTrajectory")
#   num_timesteps = trajectory.num_timesteps
#   value_targets = np.zeros([num_timesteps])
#   value_targets[-1] = trajectory.rewards[-1]
#   if not trajectory.resets[-1]:
#     obs = trajectory.latest_observation
#     value_targets[-1] += gamma * policy.act(obs, sess)[1]
#   for i in reversed(range(num_timesteps-1)):
#     not_reset = 1 - trajectory.resets[i]
#     value_targets[i] = trajectory.rewards[i]\
#         + not_reset * gamma * value_targets[i+1]
#   advantages = value_targets - trajectory.value_preds
#   return advantages, value_targets


def gae(policy, trajectory, gamma=0.99, lambda_=0.95, sess=None):
  if not isinstance(policy, rl.policies.ValueFunctionPolicy):
    raise TypeError("policy must be an instance of ValueFunctionPolicy")
  if not isinstance(trajectory, ValueFunctionTrajectory):
    raise TypeError("trajectory must be an instance of ValueFunctionTrajectory")
  num_timesteps = trajectory.num_timesteps
  gae = np.zeros([num_timesteps])
  gae[-1] = trajectory.rewards[-1] - trajectory.value_preds[-1]
  if not trajectory.resets[-1]:
    obs = trajectory.latest_observation
    gae[-1] += gamma * policy.act(obs, sess)[1]
  for i in reversed(range(num_timesteps-1)):
    not_reset = 1 - trajectory.resets[i] # i is for next state
    delta = trajectory.rewards[i]\
        + not_reset * gamma * trajectory.value_preds[i+1]\
        - trajectory.value_preds[i]
    gae[i] = delta + not_reset * gamma * lambda_ * gae[i+1]
  value_targets = gae + trajectory.value_preds
  return gae, value_targets


class A2CTrainer(object):

  def __init__(self,
               env,
               policy, *,
               trajectory_length,
               entropy_coef=0.01,
               value_loss_coef=0.25,
               name=None):
    self.policy = policy
    self.trajectory = ValueFunctionTrajectory(env, policy, trajectory_length)
    if name is None:
      name = self.__class__.__name__
    with tf.variable_scope(None, name) as scope:
      self.scope = scope
      self.global_step = tf.train.create_global_step()
      self.actions = tf.placeholder(self.policy.distribution.dtype,
                                    [None], name="actions")
      self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
      self.value_targets = tf.placeholder(tf.float32, [None],
                                          name="value_targets")
      with tf.name_scope("loss"):
        pd = self.policy.distribution
        self.policy_loss = tf.reduce_sum(
            pd.neglogp(self.actions) * self.advantages\
              - entropy_coef * tf.reduce_mean(pd.entropy()))
        self.v_loss = tf.reduce_sum(tf.square(
            tf.squeeze(self.policy.value_preds) - self.value_targets))
        self.loss = self.policy_loss + value_loss_coef * self.v_loss
        self.gradients = tf.gradients(self.loss, self.policy.var_list())
        self.grads_and_vars = zip(
            self.policy.gradient_preprocessing(self.gradients),
            self.policy.var_list()
          )
      self._init_summaries()

  def _init_summaries(self):
    with tf.variable_scope("summaries") as scope:
      self.episode_length = tf.placeholder(tf.int64, name="episode_length_ph")
      self.episode_reward = tf.placeholder(tf.float32,
                                           name="episode_reward_ph")
      tf.summary.scalar("value_preds",
                        tf.reduce_mean(self.policy.value_preds))
      tf.summary.scalar("value_targets",
                        tf.reduce_mean(self.value_targets))
      tf.summary.scalar("distribution_entropy",
                        tf.reduce_mean(self.policy.distribution.entropy()))
      batch_size = tf.to_float(tf.shape(self.actions)[0])
      tf.summary.scalar("policy_loss", self.policy_loss / batch_size)
      tf.summary.scalar("value_loss", self.v_loss / batch_size)
      tf.summary.scalar("loss", self.loss / batch_size)
      tf.summary.scalar("gradient_norm", tf.global_norm(self.gradients))
      tf.summary.scalar("policy_norm", tf.global_norm(self.policy.var_list()))
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name)
      self.summaries = tf.summary.merge(summaries)

  def train(self,
            num_steps,
            optimizer,
            logdir,
            gamma=0.99,
            lambda_=0.95,
            summary_period=10,
            checkpoint_period=100,
            checkpoint=None):
    train_op = tf.group(optimizer.apply_gradients(self.grads_and_vars),
                        self.global_step.assign_add(1))
    with tf.Session() as sess, sess.as_default():
      summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
      saver = tf.train.Saver()
      save_path = os.path.join(logdir, "checkpoint")
      if checkpoint is not None:
        saver.restore(sess, checkpoint)
      else:
        tf.global_variables_initializer().run()

      try:
        while self.global_step.eval() < num_steps:
          i = self.global_step.eval()
          self.trajectory.update(summary_writer, sess=sess)
          advantages, value_targets = gae(
              self.policy, self.trajectory,
              gamma=gamma, lambda_=lambda_, sess=sess)
          feed_dict = {
              self.policy.inputs: self.trajectory.observations,
              self.actions: self.trajectory.actions,
              self.advantages: advantages,
              self.value_targets: value_targets
          }
          if self.policy.get_state() is not None:
            feed_dict[self.policy.state_in] = self.trajectory.policy_states[0]
          if i % summary_period == 0:
            fetches = [self.loss,
                       self.policy_loss,
                       self.v_loss,
                       train_op,
                       self.summaries]
            loss, policy_loss, v_loss, _, summaries =\
                sess.run(fetches, feed_dict)
            summary_writer.add_summary(summaries, global_step=i)
            msg = "Training step #{}:  "\
                  "Loss: {:.4}, Policy loss: {:.4}, Value loss: {:.4f}"\
                  .format(i, loss, policy_loss, v_loss)
            logging.info(msg)
          else:
            train_op.run(feed_dict)
          if i > 0 and i % checkpoint_period == 0:
            saver.save(sess, save_path, global_step=i)
            logging.info("### Model {} Saved ###".format(i))
      finally:
        saver.save(sess, save_path, global_step=self.global_step.eval())
        logging.info("### Model {} Saved ###".format(self.global_step.eval()))
