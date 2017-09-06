import copy
import logging
import threading

import numpy as np
import tensorflow as tf

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
        if (step - self.last_summary_step) >= self.summary_period:
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
    with self.sess.graph.as_default():
      while not self.sess.should_stop():
        self.trajectory = copy.deepcopy(self.trajectory)
        self.rollout()
        self.queue.put(self.trajectory)


class GAE(object):
  def __init__(self, policy, gamma=0.99, lambda_=0.95):
    self._policy = policy
    self._gamma = gamma
    self._lambda_ = lambda_

  def __call__(self, trajectory, sess=None):
    num_timesteps = trajectory.num_timesteps
    gae = np.zeros([num_timesteps])
    gae[-1] = trajectory.rewards[num_timesteps-1]\
        - trajectory.value_preds[num_timesteps-1]
    if not trajectory.resets[num_timesteps-1]:
      obs = trajectory.latest_observation
      gae[-1] += self._gamma * self._policy.act(obs, sess)[1]
    for i in reversed(range(num_timesteps-1)):
      not_reset = 1 - trajectory.resets[i] # i is for next state
      delta = trajectory.rewards[i]\
          + not_reset * self._gamma * trajectory.value_preds[i+1]\
          - trajectory.value_preds[i]
      gae[i] = delta + not_reset * self._gamma * self._lambda_ * gae[i+1]
    value_targets = gae + trajectory.value_preds[:num_timesteps]
    return gae, value_targets
