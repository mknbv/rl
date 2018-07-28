from abc import ABC, abstractmethod
from collections import namedtuple
from logging import getLogger
from operator import attrgetter
import pickle

import numpy as np
import tensorflow as tf

from .interactions_producer import BaseInteractionsProducer

USE_DEFAULT = object()
logger = getLogger("rl")


class BaseExperienceStorage(ABC):
  def __init__(self, capacity):
    self._capacity = capacity
    self._storage = np.empty(capacity, dtype=np.object)
    self._index = 0
    self._filled = False

  @property
  def capacity(self):
    return self._capacity

  @property
  def filled(self):
    return self._filled

  @property
  def size(self):
    return self._index if not self.filled else self.capacity

  @classmethod
  def fromfile(cls, filename):
    logger.info("Loading experience from {}".format(filename))
    with open(filename, "rb") as picklefile:
      return pickle.load(picklefile)

  def save(self, filename):
    logger.info("Saving experience to {}".format(filename))
    with open(filename, "wb") as picklefile:
      pickle.dump(self, picklefile)

  def put(self, data):
    self._storage[self._index] = data
    self._index += 1
    if self._index == self.capacity:
      if not self._filled:
        self._filled = True
      self._index = 0

  def get(self, indices):
    if not self.filled and np.any(self._index <= indices):
      raise ValueError("indices exceed effective size")
    data = self._storage[indices]
    return data

  @abstractmethod
  def sample(self, sample_size):
    ...

  def latest(self):
    if self._index == 0 and not self.filled:
      raise ValueError("experience is empty")
    return self._storage[(self._index + self.capacity - 1) % self.capacity]


class UniformSamplerStorage(BaseExperienceStorage):
  def sample(self, sample_size):
    if self.size == 0:
      raise ValueError("cannot sample from empty storage")
    indices = np.random.choice(self.size, size=sample_size)
    return {"data": self.get(indices)}


class _SumTree(object):
  def __init__(self, size):
    self.size = size
    self.data = np.zeros(2 * self.size - 1)

  @property
  def total_sum(self):
    return self.data[0]

  def val(self, index):
    return self.data[index + self.size - 1]

  def add(self, index, val):
    index += self.size - 1
    self.data[index] += val
    while index:
      index = index - 1 >> 1
      self.data[index] += val

  def replace(self, index, val):
    index += self.size - 1
    old_val = self.data[index]
    self.data[index] = val
    while index:
      index = index - 1 >> 1
      self.data[index] = self.data[index] - old_val + val

  def retrieve(self, val):
    index = 0
    while index < self.size - 1:
      left_index = 2 * index + 1
      right_index = 2 * index + 2
      if self.data[left_index] >= val:
        index = left_index
      else:
        val -= self.data[left_index]
        index = right_index
    return index - self.size + 1


class PrioritizedSamplerStorage(BaseExperienceStorage):
  def __init__(self, capacity, start_max_priority=1):
    super(PrioritizedSamplerStorage, self).__init__(capacity)
    self._sum_tree = _SumTree(capacity)
    self._max_priority = start_max_priority

  def put(self, data, priority=None):
    if priority is None:
      priority = self._max_priority
    else:
      self._max_priority = max(self._max_priority, priority)
    self._sum_tree.replace(self._index, priority)
    super(PrioritizedSamplerStorage, self).put(data)

  def sample(self, sample_size):
    if self.size == 0:
      raise ValueError("cannot sample from empty storage")
    max_sums = np.linspace(0, self._sum_tree.total_sum, sample_size+1)
    sample_sums = np.random.uniform(max_sums[:-1], max_sums[1:])
    indices = np.asarray([self._sum_tree.retrieve(s) for s in sample_sums])
    sample = {"data": self.get(indices)}
    sample["indices"] = indices
    sample["log_probs"] = (np.log(self._sum_tree.val(indices))
                           - np.log(self._sum_tree.total_sum))
    return sample

  def update_priorities(self, indices, priorities):
    for ind, pr in zip(indices, priorities):
      if not self.filled and ind > self._index:
        raise ValueError("index value {} is outside of the storage"
                         .format(ind))
      self._sum_tree.replace(ind, pr)


class ExperienceTuple(namedtuple("ExperienceTuple",
                                 ["observation", "action", "reward",
                                  "done", "next_observation"])):
  @staticmethod
  def tuples_to_batches(experience_tuples):
    return list(map(np.array, zip(*experience_tuples)))


class BaseExperienceReplay(BaseInteractionsProducer):
  def __init__(self, env, policy, storage_class,
               experience_size,
               experience_start_size,
               batch_size=32,
               steps_per_next=4,
               env_step=None):
    super(BaseExperienceReplay, self).__init__(env, policy, batch_size,
                                               env_step)
    self._storage_class = storage_class
    self._storage = None
    self._experience_size = experience_size
    self._experience_start_size = experience_start_size
    self._last_checkpoint_step = None
    self._steps_per_next = steps_per_next

  @property
  def storage(self):
    return self._storage

  def restore_experience(self, fname):
    logger.info("Restoring experience from {}".format(fname))
    self._storage = self._storage_class.fromfile(fname)

  def start(self, session, summary_manager=None):
    super(BaseExperienceReplay, self).start(session, summary_manager)
    self._latest_observation = self._env.reset()
    if self._storage is not None:
      # Experience was restored.
      return

    self._storage = self._storage_class(self._experience_size)
    logger.info("Initializing experience replay")
    while self._storage.size < self._experience_start_size:
      ob = self._latest_observation
      action = self.action_space.sample()
      self._latest_observation, rew, done, _ = self._env.step(action)
      if done:
        self._latest_observation = self._env.reset()
      experience_tuple = ExperienceTuple(ob, action, rew, done,
                                         self._latest_observation)
      self._storage.put(experience_tuple)

  def next(self):
    for i in range(self._steps_per_next):
      ob = self._latest_observation
      action = self._policy.get_single_action(ob, self._session)
      self._latest_observation, rew, done, info = self._env.step(action)
      if done:
        self._latest_observation = self._env.reset()
        if self.summary_manager is not None:
          env_step = self._session.run(self.env_step) + i
          if self.summary_manager.summary_time(step=env_step):
            self.summary_manager.add_summary_dict(
                info.get("summaries", info), step=env_step)
      experience_tuple = ExperienceTuple(ob, action, rew, done,
                                         self._latest_observation)
      self._storage.put(experience_tuple)

    sample = self._storage.sample(self._batch_size)
    sample.update(zip(["observations", "actions", "rewards",
                       "resets", "next_observations"],
                      ExperienceTuple.tuples_to_batches(sample["data"])))
    sample["env_steps"] = self._steps_per_next
    self._update_env_step(sample["env_steps"])
    return sample


class UniformExperienceReplay(BaseExperienceReplay):
  def __init__(self, env, policy, experience_size, experience_start_size,
               batch_size=32,
               steps_per_next=4,
               env_step=None):
    super(UniformExperienceReplay, self).__init__(
        env, policy,
        storage_class=UniformSamplerStorage,
        experience_size=experience_size,
        experience_start_size=experience_start_size,
        batch_size=batch_size,
        steps_per_next=steps_per_next,
        env_step=env_step)


class PrioritizedExperienceReplay(BaseExperienceReplay):
  def __init__(self, env, policy, experience_size, experience_start_size,
               alpha=0.6,
               beta=USE_DEFAULT,
               epsilon=1e-8,
               batch_size=32,
               steps_per_next=4,
               env_step=None):
    super(PrioritizedExperienceReplay, self).__init__(
        env, policy,
        storage_class=PrioritizedSamplerStorage,
        experience_size=experience_size,
        experience_start_size=experience_start_size,
        batch_size=batch_size,
        steps_per_next=steps_per_next,
        env_step=env_step)
    self._alpha = alpha
    if beta == USE_DEFAULT:
      beta = PrioritizedExperienceReplay.beta(
          start_beta=0.4, end_beta=1,
          beta_anneal_steps=int(200e6), step=self.env_step)
    self._beta = beta
    self._epsilon = epsilon
    self._errors = np.zeros(experience_size, dtype=np.float32)

  @staticmethod
  def beta(start_beta, end_beta, beta_anneal_steps, step):
    start_beta = tf.constant(start_beta, dtype=tf.float32)
    end_beta = tf.constant(end_beta, dtype=tf.float32)
    step = tf.cast(tf.minimum(step, beta_anneal_steps), tf.float32)
    return (start_beta - end_beta) * (1. - step / beta_anneal_steps) + end_beta

  def next(self):
    sample = super(PrioritizedExperienceReplay, self).next()
    beta = self._session.run(self._beta)
    log_weights = -beta * (np.log(self._storage.size) + sample["log_probs"])
    sample["weights"] = np.exp(log_weights - np.max(log_weights))
    return sample

  def update_priorities(self, indices, errors):
    prev_indices = ((indices + self.storage.capacity - 1)
                    % self.storage.capacity)
    if not self._storage.filled:
      mask = indices != 0
    else:
      mask = np.ones(indices.shape[0], dtype=np.bool)
    prev_resets = np.asarray(list(map(attrgetter("done"),
                             self._storage.get(prev_indices[mask]))))
    mask[mask] = ~prev_resets
    prev_indices = prev_indices[mask]
    indices = np.hstack([prev_indices, indices])
    errors = np.hstack([self._errors[prev_indices] + errors[mask],
                        errors])
    self._storage.update_priorities(
      indices, np.power(errors + self._epsilon, self._alpha))
    self._errors[indices] = errors
