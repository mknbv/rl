import cv2
from datetime import datetime
import gym
from gym.envs.atari import AtariEnv
import gym.spaces as spaces
import numpy as np


class ImageCropping(gym.ObservationWrapper):
  def __init__(self, env, offset_height, offset_width,
               target_height, target_width):
    super(ImageCropping, self).__init__(env)
    self.offset_height = offset_height
    self.offset_width = offset_width
    self.target_height = target_height
    self.target_width = target_width

  def _crop(self, obs):
    return obs[self.offset_height:self.offset_height+self.target_height,
               self.offset_width:self.offset_width+self.target_width, :]

  def _step(self, action):
    obs, rew, done, info = self.env.step(action)
    return self._crop(obs), rew, done, info

  def _reset(self):
    return self._crop(self.env.reset())


class ImagePreprocessing(gym.ObservationWrapper):
  def __init__(self, env, grayscale):
    super(ImagePreprocessing, self).__init__(env)
    self._shape = shape
    self._grayscale = grayscale
    if self._grayscale:
      self.observation_space = spaces.Box(low=0, high=1, shape=shape)
    else:
      obs_shape = list(shape) + list(self.observation_space.shape[2:])
      self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape)

  def _preprocess(self, frame):
    preprocessed = cv2.resize(frame, self._shape)
    if self._grayscale:
      preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2GRAY)
      preprocessed = preprocessed.astype(np.float32) / 255.0
    return preprocessed

  def _step(self, action):
    obs, reward, done, info = self.env.step(action)
    obs = self._preprocess(obs)
    return obs, reward, done, info

  def _reset(self):
    obs = self.env.reset()
    return self._preprocess(obs)


class UniverseStarter(gym.ObservationWrapper):
  def __init__(self, env, keepdims=True):
    if not isinstance(env.unwrapped, AtariEnv):
      raise TypeError("env must be an AtariEnv")
    super(UniverseStarter, self).__init__(env)
    self.env = ImageCropping(env, 34, 0, 160, 160)
    self._keepdims = keepdims
    obs_shape = (42, 42) + ((1,) if self._keepdims else ())
    self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape)

  def _preprocess_observation(self, obs):
    obs = cv2.resize(obs, (80, 80))
    obs = cv2.resize(obs, (42, 42))
    obs = np.mean(obs, axis=-1, keepdims=self._keepdims) / 255.0
    return obs

  def _step(self, action):
    obs, rew, done, info = self.env.step(action)
    obs = self._preprocess_observation(obs)
    return obs, rew, done, info

  def _reset(self):
    return self._preprocess_observation(self.env.reset())


class MaxBetweenFrames(gym.ObservationWrapper):
  def __init__(self, env):
    super(MaxBetweenFrames, self).__init__(env)
    self._last_obs = None

  def _step(self, action):
    true_obs, reward, done, info = self.env.step(action)
    obs = np.maximum(true_obs, self._last_obs)
    self._last_obs = true_obs
    return obs, reward, done, info

  def _reset(self):
    self._last_obs = self.env.reset()
    return self._last_obs


class QueueFrames(gym.ObservationWrapper):
  def __init__(self, env, num_frames):
    super(QueueFrames, self).__init__(env)
    self._num_frames = num_frames
    self._obs_queue = None
    obs_shape = list(self.observation_space.shape) + [num_frames]
    self.observation_space = spaces.Box(
        low=self.observation_space.low.min(),
        high=self.observation_space.high.max(),
        shape=obs_shape)

  def _reset_obs_queue(self):
    obs = self.env.reset()
    obs_queue = np.empty(obs.shape + (self._num_frames,))
    for i in range(self._num_frames-1):
      obs_queue[..., i] = np.zeros_like(obs)
    obs_queue[..., -1] = obs
    return obs_queue

  def _step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._obs_queue = np.append(self._obs_queue[..., 1:],
                                np.expand_dims(obs, -1), axis=-1)
    return self._obs_queue, reward, done, info

  def _reset(self):
    self._obs_queue = self._reset_obs_queue()
    return self._obs_queue


class ClipReward(gym.RewardWrapper):
  def _reward(self, reward):
    return np.sign(reward)


class Logging(gym.Wrapper):
  def __init__(self, env):
    super(Logging, self).__init__(env)
    self._episode_counter = 0

  def _step(self, action):
    if self.first_step:
      self.start_time = datetime.now()
      self.first_step = False
    obs, rew, done, info = self.env.step(action)
    self.total_reward += rew
    self.episode_length += 1
    info["logging.total_reward"] = self.total_reward
    info["logging.episode_length"] = self.episode_length
    interactions_per_second = None
    if done:
      delta_seconds = (datetime.now() - self.start_time).total_seconds()
      interactions_per_second = self.episode_length / delta_seconds
      self._episode_counter += 1
    info["logging.episode_counter"] = self._episode_counter
    info["logging.interactions_per_second"] = interactions_per_second
    return obs, rew, done, info

  def _reset(self):
    self.first_step = True
    self.start_time = None
    self.total_reward = 0
    self.episode_length = 0
    self.interactions_per_second = 0
    return self.env.reset()


def nature_dqn_wrap(env):
  env = MaxBetweenFrames(env)
  env = ImagePreprocessing(env, (84, 84), grayscale=True)
  env = QueueFrames(env, 4)
  env = ClipReward(env)
  return env
