import abc
import collections
import copy
import logging
import os
import threading

from gym import spaces
import numpy as np
import tensorflow as tf


__all__ = ["BaseInteractionsProducer", "OnlineInteractionsProducer"]


class _InteractionSummaryManager(object):
  def __init__(self, summary_writer, summary_period, step=None, sess=None):
    self._summary_writer = summary_writer
    self._summary_period = summary_period
    self._step = step if step is not None else tf.train.get_global_step()
    self._sess = sess or tf.get_default_session()
    self._last_summary_step = self._sess.run(self._step) - self._summary_period

  @property
  def summary_writer(self):
    return self._summary_writer

  @property
  def step(self):
    return self._step

  @property
  def step_value(self):
    return self._sess.run(self.step)

  def summary_time(self):
    return self._summary_period <= self.step_value - self._last_summary_step

  def add_summary(self, info, summaries=None, feed_dict=None):
    episode_counter = info.get("logging.episode_counter", None)
    if episode_counter is not None and "logging.total_reward" in info:
      logging.info("Episode #{} finished, reward: {}"\
                   .format(episode_counter, info["logging.total_reward"]))
    if summaries is not None:
      fetched_summaries = self._sess.run(summaries, feed_dict)
      self.summary_writer.add_summary(fetched_summaries, self.step_value)
    summary = tf.Summary()
    logkeys = filter(
        lambda k: k.startswith("logging") and k != "logging.episode_counter",
        info.keys()
    )
    for key in logkeys:
      val = float(info[key])
      tag = "Trajectory/" + key.split(".", 1)[1]
      summary.value.add(tag=tag, simple_value=val)
    self._summary_writer.add_summary(summary, self.step_value)
    self._last_summary_step = self.step_value


class BaseInteractionsProducer(abc.ABC):
  def __init__(self, env, policy, batch_size):
    self._env = env
    self._policy = policy
    self._batch_size = batch_size

  @property
  def observation_space(self):
    return self._env.observation_space

  @property
  def action_space(self):
    return self._env.action_space

  @abc.abstractmethod
  def start(self, summary_writer, summary_period, sess=None):
    if not self._policy.is_built:
      raise ValueError("Policy must be built before calling start")
    self._sess = sess or tf.get_default_session()
    self._summary_manager = _InteractionSummaryManager(
        summary_writer=summary_writer,
        summary_period=summary_period,
        sess=self._sess)

  @abc.abstractmethod
  def next(self):
    ...


class OnlineInteractionsProducer(BaseInteractionsProducer):
  def __init__(self, env, policy, batch_size, queue=None):
    super(OnlineInteractionsProducer, self).__init__(env, policy, batch_size)
    self._queue = queue

    if self._policy.metadata.get("visualize_observations", False):
      # We cannot create tensor with this summary inside of `start` call,
      # since at that time the `tf.Graph` might already be finilized.
      def _set_summaries(built_policy, *args, **kwargs):
        self._summaries = tf.summary.image("Trajectory/observation",
                                           built_policy.observations)
      self._policy.add_after_build_hook(_set_summaries)
    else:
      self._summaries = None


  def start(self, summary_writer, summary_period=500, sess=None):
    super(OnlineInteractionsProducer, self).start(
        summary_writer=summary_writer,
        summary_period=summary_period,
        sess=sess)

    latest_observation = self._env.reset()
    obs_shape = (self._batch_size,) + latest_observation.shape
    obs_type = latest_observation.dtype
    if isinstance(self._env.action_space, spaces.Discrete):
      act_shape = tuple()
    else:
      act_shape = self._env.action_space.shape
    act_type = np.asarray(self.action_space.sample()).dtype
    act_shape = (self._batch_size,) + act_shape
    self._trajectory = {
        "latest_observation": latest_observation,
        "observations": np.empty(obs_shape, dtype=obs_type),
        "actions": np.empty(act_shape, dtype=act_type),
        "rewards": np.empty(self._batch_size, dtype=np.float32),
        "resets": np.empty(self._batch_size, dtype=np.bool),
        "critic_values": np.empty(self._batch_size, dtype=np.float32),
        "num_timesteps": self._batch_size,
    }
    if self._policy.state_values is not None:
      self._trajectory["policy_state"] = self._policy.state_values

    # Launch policy.
    if self._queue is not None:
      thread = threading.Thread(target=self._feed_queue, daemon=True)
      logging.info("Launching daemon agent")
      thread.start()

  def rollout(self):
    traj = self._trajectory
    observations = traj["observations"]
    actions = traj["actions"]
    rewards = traj["rewards"]
    resets = traj["resets"]
    critic_values = traj["critic_values"]
    traj["num_timesteps"] = self._batch_size
    if "policy_state" in traj:
      traj["policy_state"] = self._policy.state_values
    for i in range(self._batch_size):
      observations[i] = self._trajectory["latest_observation"]
      actions[i], critic_values[i] =\
          self._policy.act(traj["latest_observation"], sess=self._sess)
      traj["latest_observation"], rewards[i], resets[i], info =\
          self._env.step(actions[i])
      if resets[i]:
        traj["latest_observation"] = self._env.reset()
        self._policy.reset()
        if self._summary_manager.summary_time():
          self._add_summary(info)
        # Recurrent policies require trajectory to end when episode ends.
        # Otherwise the batch may combine interactions from differen episodes.
        if self._policy.state_inputs is not None:
          traj["num_timesteps"] = i + 1
          break

  def next(self):
    if self._queue is not None:
      traj = self._queue.get()
      return traj
    self.rollout()
    return self._trajectory

  def _add_summary(self, info):
    feed_dict = None
    if self._summaries is not None:
      feed_dict = {self._policy.observations: self._trajectory["observations"]}
    self._summary_manager.add_summary(
        info, summaries=self._summaries, feed_dict=feed_dict)
    if self._queue is not None:
      summary = tf.Summary()
      tag = "Trajectory/queue_fraction_of_{}_full".format(self._queue.maxsize)
      frac = self._queue.qsize() / self._queue.maxsize
      summary.value.add(tag=tag, simple_value=frac)
      self._summary_manager.summary_writer.add_summary(
          summary, self._summary_manager.step_value)

  def _feed_queue(self):
    assert self._queue is not None
    with self._sess.graph.as_default():
      while not self._sess.should_stop():
        self._trajectory = copy.deepcopy(self._trajectory)
        self.rollout()
        self._queue.put(self._trajectory)
