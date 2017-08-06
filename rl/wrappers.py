from datetime import datetime
import gym
from gym.envs.atari import AtariEnv
import gym.spaces as spaces
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray


def wrap(env, wrappers):
  for wrap in wrappers:
    env = wrap(env)
  return env


def ImageCroppingWrapper(offset_height, offset_width,
                         target_height, target_width):
  class ImageCroppingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
      super(ImageCroppingWrapper, self).__init__(env)
      self.offset_height = offset_height
      self.offset_width = offset_width
      self.target_height = target_height
      self.target_width = target_width

    def _crop(self, obs):
      return obs[self.offset_height:self.offset_height+self.target_height,
                 self.offset_width:self.offset_width+self.target_width, :]

    def _step(self, action):
      obs, rew, done, info = self.env.step(action)
      info["image.cropping.offset.height"] = self.offset_height
      info["image.cropping.offset.width"] = self.offset_width
      info["image.cropping.target.height"] = self.target_height
      info["image.cropping.target.width"] = self.target_width
      return self._crop(obs), rew, done, info

    def _reset(self):
      return self._crop(self.env.reset())
  return ImageCroppingWrapper

def ImagePreprocessingWrapper(shape, grayscale):
  class ImagePreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
      super(ImagePreprocessingWrapper, self).__init__(env)
      self._shape = shape
      self._grayscale = grayscale
      if self._grayscale:
        self.observation_space = spaces.Box(low=0, high=1, shape=shape)
      else:
        obs_shape = list(shape) + list(self.observation_space.shape[2:])
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape)

    def _preprocess(self, frame):
      preprocessed = resize(frame, self._shape, mode="constant")
      if self._grayscale:
        preprocessed = rgb2gray(preprocessed)
      return preprocessed

    def _step(self, action):
      obs, reward, done, info = self.env.step(action)
      obs = self._preprocess(obs)
      info["image.preprocessing"] = {
          "shape": self._shape,
          "grayscale": self._grayscale
      }
      return obs, reward, done, info

    def _reset(self):
      obs = self.env.reset()
      return self._preprocess(obs)

  return ImagePreprocessingWrapper


def UniverseStarterImageWrapper():
  class UniverseStarterImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
      if not isinstance(env.unwrapped, AtariEnv):
        raise TypeError("env must be an AtariEnv")
      super(UniverseStarterImageWrapper, self).__init__(env)
      self.env = ImageCroppingWrapper(34, 0, 159, 159)(env)
      self.observation_space = spaces.Box(low=0, high=1, shape=(42, 42, 1))

    def _preprocess_observation(self, obs):
      obs = resize(obs, (80, 80), mode="constant")
      obs = resize(obs, (42, 42), mode="constant")
      obs = np.mean(obs, axis=-1, keepdims=True) / 255.0
      return obs

    def _step(self, action):
      obs, rew, done, info = self.env.step(action)
      obs = self._preprocess_observation(obs)
      return obs, rew, done, info

    def _reset(self):
      return self._preprocess_observation(self.env.reset())

  return UniverseStarterImageWrapper


def MaxBetweenFramesWrapper():
  class MaxBetweenFramesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
      super(MaxBetweenFramesWrapper, self).__init__(env)
      self._last_obs = None

    def _step(self, action):
      true_obs, reward, done, info = self.env.step(action)
      obs = np.max([true_obs, self._last_obs], axis=0)
      self._last_obs = true_obs
      if "max.between.observations" in info:
        raise gym.error.Error(
            "Key 'max.between.observations' already in "\
            "info. Make sure you are not stacking the "\
            "MaxBetweenObservationsWrapper wrappers.")
      info["max.between.observations"] = ""
      return obs, reward, done, info

    def _reset(self):
      self._last_obs = self.env.reset()
      return self._last_obs

  return MaxBetweenFramesWrapper


def QueueFramesWrapper(num_frames):
  class QueueFramesWrapper(gym.ObservationWrapper):
    def __init__(self, env):
      super(QueueFramesWrapper, self).__init__(env)
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
      if "queue.num_frames" in info:
        raise gym.error.Error("Key 'queue.num_frames' already in info. Make "\
              "sure you are not stacking the QueueFramesWrapper wrappers.")
      info["queue.num_frames"] = self._num_frames
      return self._obs_queue, reward, done, info

    def _reset(self):
      self._obs_queue = self._reset_obs_queue()
      return self._obs_queue

  return QueueFramesWrapper


def ClipRewardWrapper():
  class ClipRewardWrapper(gym.RewardWrapper):
    def _reward(self, reward):
      return np.sign(reward)
  return ClipRewardWrapper


def LoggingWrapper():
  class LoggingWrapper(gym.Wrapper):

    def _step(self, action):
      if self.first_step:
        self.start_time = datetime.now()
        self.first_step = False
      obs, rew, done, info = self.env.step(action)
      if "logging.total_reward" in info:
        raise ValueError("Key 'logging.total_reward' already in info. Make "\
            "sure you are not stacking the LoggingWrapper wrappers.")
      self.total_reward += rew
      self.episode_length += 1
      info["logging.total_reward"] = self.total_reward
      info["logging.episode_length"] = self.episode_length
      interactions_per_second = None
      if done:
        delta_seconds = (datetime.now() - self.start_time).total_seconds()
        interactions_per_second = self.episode_length / delta_seconds
      info["logging.interactions_per_second"] = interactions_per_second
      return obs, rew, done, info

    def _reset(self):
      self.first_step = True
      self.start_time = None
      self.total_reward = 0
      self.episode_length = 0
      self.interactions_per_second = 0
      return self.env.reset()

  return LoggingWrapper


def NatureDQNWrapper():
  wrappers = [
      ImagePreprocessingWrapper((84, 84), grayscale=True),
      MaxBetweenFramesWrapper(),
      QueueFramesWrapper(4),
      ClipRewardWrapper()
  ]
  return lambda env: wrap(env, wrappers)
