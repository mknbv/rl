from collections import namedtuple
import copy
from datetime import datetime
import logging
import numpy as np
import os
import tensorflow as tf
import threading
import queue

import rl.policies


class Trajectory(object):
  def __init__(self, initial_obs, act_shape, act_type, num_timesteps,
               record_policy_states=False):
    self.latest_observation = initial_obs
    self.num_timesteps = num_timesteps
    obs_shape = list(self.latest_observation.shape)
    obs_type = self.latest_observation.dtype
    self.observations = np.zeros([num_timesteps] + obs_shape, dtype=obs_type)
    self.resets = np.zeros([num_timesteps], dtype=np.bool)
    self.rewards = np.zeros([num_timesteps])
    self.actions = np.zeros([num_timesteps] + act_shape, dtype=act_type)
    self.value_preds = np.zeros([self.num_timesteps], dtype=np.float32)
    if record_policy_states:
      self.policy_states = [None] * self.num_timesteps
    else:
      self.policy_states = None


class TrajectoryProducer(object):
  def __init__(self, env, policy, num_timesteps, queue=None):
    self.env = env
    self.policy = policy
    self.act_shape = list(self.policy.distribution.shape[1:])
    self.act_type = self.policy.distribution.dtype.as_numpy_dtype
    self.num_timesteps = num_timesteps
    policy_is_recurrent = self.policy.get_state() is not None
    self.trajectory = Trajectory(
        env.reset(), self.act_shape, self.act_type, self.num_timesteps,
        record_policy_states=policy_is_recurrent)
    self.episode_count = 1
    self.last_summary_step = 1
    self.queue = queue
    self.hard_cutoff = policy_is_recurrent
    self.summary_writer = None
    self.summary_period = None
    self.sess = None

    if isinstance(self.policy, rl.policies.CNNPolicy):
      self.summaries = tf.summary.image("Trajectory/observation",
                                        self.policy.inputs)
    else:
      self.summaries = None

  def start(self, summary_writer, summary_period=500, sess=None):
    self.summary_writer = summary_writer
    self.summary_period = summary_period
    self.sess = sess or tf.get_default_session()
    if self.queue is not None:
      thread = threading.Thread(target=self._feed_queue, daemon=True)
      logging.info("Launching daemon agent")
      thread.start()

  def rollout(self):
    traj = self.trajectory
    traj.num_timesteps = self.num_timesteps
    for i in range(self.num_timesteps):
      traj.observations[i] = traj.latest_observation
      if traj.policy_states is not None:
        traj.policy_states[i] = self.policy.get_state()
      traj.actions[i], traj.value_preds[i] =\
          self.policy.act(traj.latest_observation, sess=self.sess)
      traj.latest_observation, traj.rewards[i], traj.resets[i], info =\
          self.env.step(traj.actions[i])
      if traj.resets[i]:
        traj.latest_observation = self.env.reset()
        self.policy.reset()
        step = self.sess.run(tf.train.get_global_step())
        if (step - self.last_summary_step) > self.summary_period:
          self._add_summary(info, self.summary_writer, step, sess=self.sess)
          self.last_summary_step = step
        self.episode_count += 1
        if self.hard_cutoff:
          traj.num_timesteps = i + 1
          break

  def next(self):
    if self.queue is not None:
      traj = self.queue.get()
      return traj
    self.rollout()
    return self.trajectory

  def _add_summary(self, info, summary_writer, step, sess):
    summary = tf.Summary()
    with tf.variable_scope(None, self.__class__.__name__):
      logkeys = filter(lambda k: k.startswith("logging"), info.keys())
      for key in logkeys:
        val = float(info[key])
        tag = "Trajectory/" + key.split(".", 1)[1]
        summary.value.add(tag=tag, simple_value=val)
      if self.queue is not None:
        tag = "Trajectory/queue_fraction_of_{}_full".format(self.queue.maxsize)
        summary.value.add(
            tag=tag, simple_value=self.queue.qsize() / self.queue.maxsize)
    logging.info("Episode #{} finished, reward: {}"\
        .format(self.episode_count, info["logging.total_reward"]))
    summary_writer.add_summary(summary, step)
    if self.summaries is not None:
      fetched_summary = sess.run(
          self.summaries, {self.policy.inputs: self.trajectory.observations})
      summary_writer.add_summary(fetched_summary, step)

  def _feed_queue(self):
    assert self.queue is not None
    with self.sess.as_default(), self.sess.graph.as_default():
      while True:
        self.trajectory = copy.deepcopy(self.trajectory)
        self.rollout()
        self.queue.put(self.trajectory)


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
  num_timesteps = trajectory.num_timesteps
  gae = np.zeros([num_timesteps])
  gae[-1] = trajectory.rewards[num_timesteps-1]\
      - trajectory.value_preds[num_timesteps-1]
  if not trajectory.resets[num_timesteps-1]:
    obs = trajectory.latest_observation
    gae[-1] += gamma * policy.act(obs, sess)[1]
  for i in reversed(range(num_timesteps-1)):
    not_reset = 1 - trajectory.resets[i] # i is for next state
    delta = trajectory.rewards[i]\
        + not_reset * gamma * trajectory.value_preds[i+1]\
        - trajectory.value_preds[i]
    gae[i] = delta + not_reset * gamma * lambda_ * gae[i+1]
  value_targets = gae + trajectory.value_preds[:num_timesteps]
  return gae, value_targets


class A2CTrainer(object):

  def __init__(self,
               env,
               policy, *,
               trajectory_length,
               queue=queue.Queue(maxsize=5),
               entropy_coef=0.01,
               value_loss_coef=0.25,
               name=None):
    self.policy = policy
    self.trajectory_producer = TrajectoryProducer(
        env, policy, trajectory_length, queue)
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
        with tf.name_scope("policy_loss"):
          self.policy_loss = tf.reduce_sum(
              pd.neglogp(self.actions) * self.advantages)
          self.policy_loss -= entropy_coef * tf.reduce_sum(pd.entropy())
        with tf.name_scope("value_loss"):
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
    train_op = tf.group(
        optimizer.apply_gradients(self.grads_and_vars),
        self.global_step.assign_add(
          tf.to_int64(tf.shape(self.policy.inputs)[0]))
      )
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=2)
    with tf.Session(config=config) as sess, sess.as_default():
      summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)
      saver = tf.train.Saver()
      save_path = os.path.join(logdir, "checkpoint")
      if checkpoint is not None:
        saver.restore(sess, checkpoint)
      else:
        tf.global_variables_initializer().run()
      self.trajectory_producer.start(summary_writer, summary_period, sess=sess)

      last_summary_step = -summary_period # always add summary on first step.
      try:
        while self.global_step.eval() < num_steps:
          i = self.global_step.eval()
          trajectory = self.trajectory_producer.next()
          advantages, value_targets = gae(
              self.policy, trajectory,
              gamma=gamma, lambda_=lambda_, sess=sess)
          feed_dict = {
              self.policy.inputs:
                  trajectory.observations[:trajectory.num_timesteps],
              self.actions: trajectory.actions[:trajectory.num_timesteps],
              self.advantages: advantages,
              self.value_targets: value_targets
          }
          if self.policy.get_state() is not None:
            feed_dict[self.policy.state_in] = trajectory.policy_states[0]
          if (i - last_summary_step) > summary_period:
            fetches = [self.loss,
                       self.policy_loss,
                       self.v_loss,
                       train_op,
                       self.summaries]
            loss, policy_loss, v_loss, _, summaries =\
                sess.run(fetches, feed_dict)
            summary_writer.add_summary(summaries, global_step=i)
            info = "Training step #{}:  "\
                  "Loss: {:.4}, Policy loss: {:.4}, Value loss: {:.4f}"\
                  .format(i, loss,
                          policy_loss / len(trajectory.observations),
                          v_loss / len(trajectory.observations))
            logging.info(info)
            last_summary_step = i
          else:
            train_op.run(feed_dict)
          if i > 0 and i % checkpoint_period == 0:
            saver.save(sess, save_path, global_step=i)
            logging.info("### Model {} Saved ###".format(i))
      finally:
        saver.save(sess, save_path, global_step=self.global_step.eval())
        logging.info("### Model {} Saved ###".format(self.global_step.eval()))
