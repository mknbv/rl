from collections import deque
from datetime import datetime

import cv2
cv2.ocl.setUseOpenCL(False)
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

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    return self._crop(obs), rew, done, info

  def reset(self):
    return self._crop(self.env.reset())


class EpisodicLife(gym.Wrapper):
  def __init__(self, env):
    super(EpisodicLife, self).__init__(env)
    self._lives = 0
    self._real_done = True

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    self._real_done = done
    info["real_done"] = done
    lives = self.env.unwrapped.ale.lives()
    if lives < self._lives and lives > 0:
      done = True
    self._lives = lives
    return obs, rew, done, info

  def reset(self):
    if self._real_done:
      obs = self.env.reset()
    else:
      obs, _, _, _ = self.env.step(0)
    self._lives = self.env.unwrapped.ale.lives()
    return obs


class FireReset(gym.Wrapper):
  def __init__(self, env):
    super(FireReset, self).__init__(env)
    if len(env.unwrapped.get_action_meanings()) < 3 or\
        env.unwrapped.get_action_meanings()[1] != "FIRE":
      raise TypeError()

  def step(self, action):
    return self.env.step(action)

  def reset(self):
    self.env.reset()
    obs, _, done, _ = self.env.step(1)
    if done:
      self.env.reset()
    obs, _, done, _ = self.env.step(2)
    if done:
      self.env.reset()
    return obs


class ImagePreprocessing(gym.ObservationWrapper):
  def __init__(self, env, shape, grayscale=True):
    super(ImagePreprocessing, self).__init__(env)
    self._shape = shape
    self._grayscale = grayscale
    if self._grayscale:
      self.observation_space = spaces.Box(low=0, high=255,
                                          shape=shape, dtype=np.uint8)
    else:
      obs_shape = shape + self.observation_space.shape[2:]
      self.observation_space = spaces.Box(low=0, high=255,
                                          shape=obs_shape, dtype=np.uint8)

  def _preprocess(self, frame):
    if self._grayscale:
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, self._shape, cv2.INTER_AREA)
    return frame

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    obs = self._preprocess(obs)
    return obs, reward, done, info

  def reset(self):
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
    self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape,
                                        dtype=np.float)

  def _preprocess_observation(self, obs):
    obs = cv2.resize(obs, (80, 80))
    obs = cv2.resize(obs, (42, 42))
    obs = np.mean(obs, axis=-1, keepdims=self._keepdims) / 255.0
    return obs

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    obs = self._preprocess_observation(obs)
    return obs, rew, done, info

  def reset(self):
    return self._preprocess_observation(self.env.reset())


class MaxBetweenFrames(gym.ObservationWrapper):
  def __init__(self, env):
    super(MaxBetweenFrames, self).__init__(env)
    if isinstance(env.unwrapped, gym.envs.atari.AtariEnv) and\
        "NoFrameskip" not in env.spec.id:
      raise TypeError("MaxBetweenFrames requires NoFrameskip in Atari env id")
    self._last_obs = None

  def step(self, action):
    true_obs, rew, done, info = self.env.step(action)
    obs = np.maximum(true_obs, self._last_obs)
    self._last_obs = true_obs
    return obs, rew, done, info

  def reset(self):
    self._last_obs = self.env.reset()
    return self._last_obs


class QueueFrames(gym.ObservationWrapper):
  def __init__(self, env, num_frames, lazy=False):
    super(QueueFrames, self).__init__(env)
    self._obs_queue = deque([], maxlen=num_frames)
    obs_shape = self.observation_space.shape + (num_frames,)
    self.observation_space.shape = obs_shape
    self._lazy = lazy

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._obs_queue.append(obs)
    if self._lazy:
      obs = LazyFrames(list(self._obs_queue))
    else:
      obs = np.dstack(list(self._obs_queue))
    return obs, reward, done, info

  def reset(self):
    obs = self.env.reset()
    for _ in range(self._obs_queue.maxlen):
      self._obs_queue.append(obs)
    if self._lazy:
      return LazyFrames(list(self._obs_queue))
    return np.dstack(list(self._obs_queue))


class LazyFrames(object):
  def __init__(self, frames):
    self._frames = frames

  def __array__(self, dtype=None):
    return np.dstack(self._frames)

  def __getitem__(self, i):
    return np.dstack(self._frames)[i]

  @property
  def dtype(self):
    return np.object

  @property
  def shape(self):
    return tuple()


class SkipFrames(gym.ObservationWrapper):
  def __init__(self, env, nskip=4):
    super(SkipFrames, self).__init__(env)
    if isinstance(env.unwrapped, gym.envs.atari.AtariEnv) and\
        "NoFrameskip" not in env.spec.id:
      raise TypeError("SkipFrames requires NoFrameskip in atari env id")
    self._nskip = nskip

  def step(self, action):
    total_reward = 0.0
    for _ in range(self._nskip):
      obs, rew, done, info = self.env.step(action)
      if done:
        break
      total_reward += rew
    return obs, total_reward, done, info

  def reset(self):
    return self.env.reset()


class ClipReward(gym.RewardWrapper):
  def __init__(self, env):
    super(ClipReward, self).__init__(env)

  def reward(self, reward):
    return np.sign(reward)


class StartWithRandomActions(gym.Wrapper, gym.envs.atari.AtariEnv):
  def __init__(self, env, max_random_actions=30):
    super(StartWithRandomActions, self).__init__(env)
    self._max_random_actions = max_random_actions
    self._real_done = True

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    # Always start with random actions if real_done is not in info.
    self._real_done = info.get("real_done", True)
    return obs, rew, done, info

  def reset(self):
    if self._real_done:
      self._real_done = False
      n = np.random.randint(self._max_random_actions+1)
      obs = self.env.reset()
      for _ in range(n):
        obs, _, _, _ = self.env.step(self.env.action_space.sample())
      return obs
    else:
      return self.env.reset()


class Logging(gym.Wrapper):
  def __init__(self, env):
    super(Logging, self).__init__(env)
    self._episode_counter = 0

  def step(self, action):
    if self.first_step:
      self.start_time = datetime.now()
      self.first_step = False
    obs, rew, done, info = self.env.step(action)
    self.total_reward += rew
    self.episode_reward += rew
    self.episode_length += 1
    info["logging.total_reward"] = self.total_reward
    info["logging.trajectory_length"] = self.episode_length
    if done:
      delta_seconds = (datetime.now() - self.start_time).total_seconds()
      interactions_per_second = self.episode_length / delta_seconds
      self._episode_counter += 1
      info["logging.episode_counter"] = self._episode_counter
      info["logging.interactions_per_second"] = interactions_per_second
    if "real_done" in info and info["real_done"]:
      info["logging.episode_reward"] = self.episode_reward
    return obs, rew, done, info

  def reset(self):
    self.first_step = True
    self.start_time = None
    self.total_reward = 0
    self.episode_reward = 0
    self.episode_length = 0
    self.interactions_per_second = 0
    return self.env.reset()


def nature_dqn_wrap(env, clip_reward=True):
  env = EpisodicLife(env)
  if "FIRE" in env.unwrapped.get_action_meanings():
    env = FireReset(env)
  env = StartWithRandomActions(env, max_random_actions=30)
  env = MaxBetweenFrames(env)
  env = SkipFrames(env, 4)
  env = ImagePreprocessing(env, (84, 84), grayscale=True)
  env = QueueFrames(env, 4, lazy=True)
  if clip_reward:
    env = ClipReward(env)
  return env
