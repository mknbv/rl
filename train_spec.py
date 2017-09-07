from gym.envs.atari import AtariEnv
import tensorflow as tf

from rl.policies import *

def create_optimizer(env, policy, global_step):
  optimizer_class = tf.train.AdamOptimizer
  if isinstance(policy, SimplePolicy):
    learning_rate = 1e-3
  elif isinstance(policy, UniverseStarterPolicy):
    learning_rate = 1e-4
  elif isinstance(policy, CNNPolicy):
    raise NotImplementedError()
  return optimizer_class(learning_rate=learning_rate)
