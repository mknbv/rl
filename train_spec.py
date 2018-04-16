import inspect

import tensorflow as tf
import gym
from gym.envs import classic_control

import rl


def _get_envs(module):
  def _predicate(type_):
    return inspect.isclass(type_) and issubclass(type_, gym.Env)
  return tuple(map(
      lambda name_value: name_value[1],
      inspect.getmembers(module, _predicate)
    )
  )

CLASSIC_CONTROL_ENVS = _get_envs(classic_control)


def _check_type(env):
  if not isinstance(env, gym.Env):
    raise TypeError("env must be instance of gym.Env: {}".format(type(env)))

def is_classic_control_env(env):
  _check_type(env)
  return isinstance(env.unwrapped, CLASSIC_CONTROL_ENVS)

def is_mujoco_env(env):
  _check_type(env)
  return isinstance(env.unwrapped, gym.envs.mujoco.mujoco_env.MujocoEnv)

def is_atari_env(env):
  _check_type(env)
  return isinstance(env.unwrapped, gym.envs.atari.AtariEnv)


def wrap_env(env, policy_class):
  if is_atari_env(env):
    if issubclass(policy_class, rl.policies.UniverseStarterPolicy):
      env = rl.env_wrappers.UniverseStarterImageWrapper(env)
      env = rl.env_wrappers.ClipRewardWrapper(env)
    elif issubclass(policy, rl.policies.CNNPolicy):
      env = rl.env_wrappers.nature_dqn_wrap(env)
  env = rl.env_wrappers.LoggingWrapper(env)
  return env

def create_policy(env, policy_class, name=None):
  kwargs = {}
  if is_classic_control_env(env) and\
      issubclass(policy_class, rl.policies.MLPPolicy):
    kwargs["num_layers"] = 2
    kwargs["units"] = 16
    kwargs["clipping_param"] = None
    kwargs["joint"] = False
  return policy_class(env.observation_space, env.action_space,
                      **kwargs, name=name)

def create_optimizer(env, policy, global_step):
  optimizer_class = tf.train.AdamOptimizer
  if is_classic_control_env(env):
    learning_rate = 1e-3
  elif isinstance(policy, rl.policies.UniverseStarterPolicy):
    learning_rate = 1e-4
  elif isinstance(policy, rl.policies.MLPPolicy):
    learning_rate = 3e-4
  else:
    raise NotImplementedError()
  return optimizer_class(learning_rate=learning_rate)

def wrap_algorithm(env, algorithm):
  return algorithm
