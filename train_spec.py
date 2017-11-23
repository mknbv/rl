from gym.envs.atari import AtariEnv
import tensorflow as tf

import rl

def create_optimizer(env, policy, global_step):
  optimizer_class = tf.train.AdamOptimizer
  if isinstance(policy, rl.policies.SimplePolicy):
    learning_rate = 1e-3
  elif isinstance(policy, rl.policies.UniverseStarterPolicy):
    learning_rate = 1e-4
  elif isinstance(policy, rl.policies.CNNPolicy):
    raise NotImplementedError()
  return optimizer_class(learning_rate=learning_rate)

def wrap_algorithm(env, algorithm):
  return algorithm
