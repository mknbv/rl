import abc

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from rl.utils import tf_utils as tfu


__all__ = [
    "BasePolicy",
    "MLPCore",
    "DQNCore",
    "UniverseStarterCore",
]

def _check_space_type(space_name, space, expected_type):
  if not isinstance(space, expected_type):
    raise ValueError(
        "{} must be an instance of gym.spaces.{}"
        .format(space_name, expected_type.__name__))


class BasePolicy(tfu.NetworkStructure):
  metadata = {}

  @abc.abstractmethod
  def __init__(self, name=None):
    self._observations = None
    self._layers = None

  @property
  def observations(self):
    return self._observations

  @property
  def layers(self):
    return self._layerse

  @property
  def state_inputs(self):
    return None

  @property
  def state_values(self):
    return None

  def reset(self):
    pass

  @abc.abstractmethod
  def act(self, observation, sess=None):
    ...

  def var_list(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             scope=self.scope.name + "/")

  def preprocess_gradients(self, grad_list):
    return grad_list


class BaseCore(abc.ABC):
  @property
  def suggested_initializer(self):
    return None

  @abc.abstractmethod
  def __call__(self):
    ...


class MLPCore(BaseCore):
  def __init__(self, num_layers, units, activation=tf.nn.tanh):
    self._num_layers = num_layers
    self._units = units
    self._activation = activation

  @property
  def suggested_initializer(self):
    return tfu.orthogonal_initializer(np.sqrt(2))

  def __call__(self):
    layers = []
    kernel_initializer = lambda scale: tfu.orthogonal_initializer(scale)
    bias_initializer = tf.zeros_initializer()
    for i in range(self._num_layers):
      if isinstance(self._units, (list, tuple)):
        layer_units = self._units[i]
      else:
        layer_units = self._units
      layers.append(
          tf.layers.Dense(
            units=layer_units,
            activation=self._activation,
            kernel_initializer=kernel_initializer(np.sqrt(2)),
            bias_initializer=bias_initializer,
            name="dense_{}".format(i)
          )
      )
    return layers


def _nips_dqn_core():
  return [
      tf.layers.Conv2D(
        filters=16,
        kernel_size=8,
        strides=4,
        activation=tf.nn.relu,
        kernel_initializer=tfu.torch_default_initializer(),
        bias_initializer=tfu.torch_default_initializer()
      ),
      tf.layers.Conv2D(
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        kernel_initializer=tfu.torch_default_initializer(),
        bias_initializer=tfu.torch_default_initializer()
      ),
      tf.layers.Flatten(),
      tfl.layers.Dense(
        units=256,
        activation=tf.nn.relu,
        kernel_initializer=tfu.torch_default_initializer(),
        bias_initializer=tfu.torch_default_initializer()
      )
  ]

def _nature_dqn_core():
  return [
      tf.layers.Conv2D(
        filters=32,
        kernel_size=8,
        strides=4,
        activation=tf.nn.relu,
        kernel_initializer=tfu.torch_default_initializer(),
        bias_initializer=tfu.torch_default_initializer()
      ),
      tf.layers.Conv2D(
        filters=64,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        kernel_initializer=tfu.torch_default_initializer(),
        bias_initializer=tfu.torch_default_initializer()
      ),
      tf.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        activation=tf.nn.relu,
        kernel_initializer=tfu.torch_default_initializer(),
        bias_initializer=tfu.torch_default_initializer()
      ),
      tf.layers.Flatten(),
      tf.layers.Dense(
        units=512,
        activation=tf.nn.relu,
        kernel_initializer=tfu.torch_default_initializer(),
        bias_initializer=tfu.torch_default_initializer()
      )
  ]

class DQNCore(BaseCore):
  def __init__(self, kind):
    if not kind in ["nature", "nips"]:
      raise TypeError("kind must be one of ['nature', 'nips']")
    self._kind = kind

  @property
  def suggested_initializer(self):
    return tfu.torch_default_initializer()

  def __call__(self):
    if self._kind == "nips":
      return _nips_dqn_core()
    else:
      return _nature_dqn_core()


class UniverseStarterCore(BaseCore):
  def __init__(self, recurrent=True):
    self._recurrent = recurrent

  def __call__(self):
    layers = []
    for i in range(4):
      layers.append(
          tf.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=tf.nn.elu,
            name="conv2d_{}".format(i + 1)
          )
      )
    layers.append(tf.layers.Flatten())
    if self._recurrent:
      layers.append(rnn.BasicLSTMCell(num_units=256))
    else:
      layers.append(tf.layers.Dense(units=256))
    return layers
