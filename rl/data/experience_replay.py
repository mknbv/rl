import logging
import pickle

from gym import spaces
import numpy as np
import tensorflow as tf

from .interactions_producer import BaseInteractionsProducer


def _observations_to_array(observations):
  observations = list(observations)
  for i, ob in enumerate(observations):
    observations[i] = np.array(ob, copy=False)
  return np.array(observations)


class Experience(object):
  def __init__(self, observation_shape, observation_type,
               action_shape, action_type, size):
    self._size = size
    self._storage = {
        "observations": np.empty((size,) + observation_shape,
                                 observation_type),
        "actions": np.empty((size,) + action_shape, action_type),
        "rewards": np.empty(size, np.float32),
        "resets": np.empty(size, np.bool),
    }
    self._index = 0
    self._filled = False

  @property
  def size(self):
    return self._size

  @property
  def filled(self):
    return self._filled

  @classmethod
  def fromfile(cls, filename):
    logging.info("Loading experience from {}".format(filename))
    with open(filename, "rb") as picklefile:
      return pickle.load(picklefile)

  def save(self, filename):
    logging.info("Saving experience to {}".format(filename))
    if not self._storage["resets"][self._index-1]:
      logging.warning(
          "Saving experience when latest observation is not at the"
          " end of an episode. This might lead to difficulties when continuing"
          " trainnig from restored experience."
      )
    with open(filename, "wb") as picklefile:
      pickle.dump(self, picklefile)

  def put(self, observation, action, reward, done):
    self._storage["observations"][self._index] = observation
    self._storage["actions"][self._index] = action
    self._storage["rewards"][self._index] = reward
    self._storage["resets"][self._index] = done

    self._index += 1
    if self._index == self.size:
      if not self._filled:
        self._filled = True
      self._index = 0

  def sample(self, sample_size):
    if self._index == 0 and not self.filled:
      raise ValueError("Cannot sample from empty experience.")
    effective_size = self.size if self.filled else self._index
    # Ignore the latest element, since we do not have the next observation for
    # it. NOTE: Technically it is possible to use this latest element
    # when its reset flag is true.
    indices = np.random.choice(effective_size-1, size=sample_size)
    if self._index > 0:
      indices[indices >= self._index - 1] += 1

    observations = self._storage["observations"][indices]
    actions = self._storage["actions"][indices]
    rewards = self._storage["rewards"][indices]
    resets = self._storage["resets"][indices]
    next_indices = (indices + 1) % effective_size
    assert not np.any(next_indices == self._index)
    next_observations = self._storage["observations"][next_indices]

    if observations.dtype == np.object:
      observations = _observations_to_array(observations)
    if next_observations.dtype == np.object:
      next_observations = _observations_to_array(next_observations)

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "resets": resets,
        "next_observations": next_observations
    }

  def latest(self):
    if self._index == 0 and not self.filled:
      raise ValueError("Experience is empty")
    if self._index == 0:
      latest_index = self._effective_size - 1
    else:
      latest_index = self._index - 1

    observation = self._storage["observations"][latest_index]
    if observation.dtype == np.object:
      observation = _observations_to_array([observation])[0]
    return {
        "observation": observation,
        "action": self._storage["actions"][latest_index],
        "reward": self._storage["rewards"][latest_index],
        "reset": self._storage["resets"][latest_index]
    }


class ExperienceReplay(BaseInteractionsProducer):
  def __init__(self,
               env, policy,
               experience_size,
               experience_start_size,
               batch_size,
               nsteps=4,
               env_step=None):
    super(ExperienceReplay, self).__init__(env, policy, batch_size,
                                           env_step=env_step)
    self._experience = None
    self._experience_size = experience_size
    self._experience_start_size = experience_start_size
    self._last_checkpoint_step = None
    self._nsteps = nsteps
    if policy.metadata.get("visualize_observations", False):
      def _set_summaries(built_policy, *args, **kwargs):
        self._summaries = tf.summary.image("Trajectory/observation",
                                           built_policy.observations)
      self._policy.add_after_build_hook(_set_summaries)
    else:
      self._summaries = None

  @property
  def experience(self):
    return self._experience

  def restore_experience(self, fname):
    logging.info("Restoring experience from {}".format(fname))
    self._experience = Experience.fromfile(fname)

  def start(self, sess, summary_manager=None):
    super(ExperienceReplay, self).start(sess, summary_manager)
    self._latest_observation = self._env.reset()
    if self._experience is not None:
      # Experience was restored.
      return

    obs_type = self._latest_observation.dtype
    obs_shape = self._latest_observation.shape
    if isinstance(self._env.action_space, spaces.Discrete):
      act_type = np.min_scalar_type(self._env.action_space.n)
      act_shape = tuple()
    else:
      act_type = self._env.action_space.sample().dtype
      act_shape = self._env.action_space.shape

    self._experience = Experience(
        obs_shape, obs_type,
        act_shape, act_type,
        self._experience_size
    )

    for _ in range(self._experience_start_size):
      obs = self._latest_observation
      action = self.action_space.sample()
      self._latest_observation, reward, done, _ = self._env.step(action)
      self._experience.put(obs, action, reward, done)
      if done:
        self._latest_observation = self._env.reset()

  def next(self):
    for i in range(self._nsteps):
      obs = self._latest_observation
      action = self._policy.act(obs[None], sess=self._session)["actions"][0]
      self._latest_observation, reward, done, info = self._env.step(action)
      self._experience.put(obs, action, reward, done)
      if done:
        self._latest_observation = self._env.reset()
        if self._summary_manager is not None:
          env_step = self._session.run(self.env_step) + i
          if self._summary_manager.summary_time(step=env_step):
            self._summary_manager.add_summary_dict(
                info.get("summaries", info), step=env_step)

    sample = self._experience.sample(self._batch_size)
    self._update_env_step(self._nsteps)
    return sample
