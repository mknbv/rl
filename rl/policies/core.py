import abc
from copy import deepcopy

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from rl.utils.tf_utils import (BuildInterface,
                               orthogonal_initializer,
                               torch_default_initializer)


def _check_space_type(space_name, space, expected_type):
  if not isinstance(space, expected_type):
    raise ValueError(
        "{} must be an instance of gym.spaces.{}"
        .format(space_name, expected_type.__name__))


class BasePolicy(BuildInterface):
  @abc.abstractmethod
  def __init__(self, name=None):
    super(BasePolicy, self).__init__(name=name)
    self._observations = None

  @property
  def observations(self):
    return self._observations

  @property
  def state_inputs(self):
    return None

  @property
  def state_values(self):
    return None

  @classmethod
  def global_and_local_instances(cls, *args, **kwargs):
    name = kwargs.get("name", cls.__name__)
    kwargs.update({"name": name + "_global"})
    global_ = cls(*args, **kwargs)
    kwargs.update({"name": name + "_local"})
    if "core" in kwargs:
      kwargs["core"] = deepcopy(kwargs["core"])
    local = cls(*args, **kwargs)
    return global_, local

  def reset(self, masks):
    pass

  @abc.abstractmethod
  def act(self, observation, sess=None):
    ...

  def var_list(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             scope=self.scope.name + "/")

  def preprocess_gradients(self, grad_list):
    return grad_list


class PolicyCore(object):
  def __init__(self, layers, kernel_initializer=None, bias_initializer=None):
    self._layers = layers
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def layers(self):
    return self._layers

  @property
  def kernel_initializer(self):
    return self._kernel_initializer

  @property
  def bias_initializer(self):
    return self._bias_initializer


class MLPCore(PolicyCore):
  def __init__(self, num_layers, units, activation=tf.nn.tanh,
               kernel_initializer=orthogonal_initializer(np.sqrt(2)),
               bias_initializer=tf.zeros_initializer()):
    layers = []
    for i in range(num_layers):
      if isinstance(units, (list, tuple)):
        layer_units = units[i]
      else:
        layer_units = units
      layers.append(
          tf.layers.Dense(
              units=layer_units,
              activation=activation,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer,
              name="dense_{}".format(i)
          )
      )
      super(MLPCore, self).__init__(layers, kernel_initializer,
                                    bias_initializer)


def layer_sequence(class_params_sequence):
  layers = []
  for layer_class, params in class_params_sequence:
    if len(params) == 0:
      layers.append(layer_class())
    else:
      values = list(params.values())
      start_len = len(values[0])
      for key, val in zip(params.keys(), values):
        if len(val) != start_len:
          raise ValueError("Layer {} parameter {} have different sizes: {}, {}"
                           .format(layer_class, key, start_len, len(val)))
      layers.extend(layer_class(**dict(zip(params.keys(), vals)))
                    for vals in zip(*list(params.values())))
  return layers


class NIPSDQNCore(PolicyCore):
  def __init__(self, kernel_initializer=torch_default_initializer(),
               bias_initializer=torch_default_initializer()):
    layers = layer_sequence([
        (tf.layers.Conv2D, {
          "filters": [16, 32],
          "kernel_size": [8, 4],
          "strides": [4, 2],
          "activation": [tf.nn.relu for _ in range(2)],
          "kernel_initializer": kernel_initializer,
          "bias_initializer": bias_initializer,
        }),
        (tf.layers.Flatten, {}),
        (tf.layers.Dense, {
            "units": [256],
            "activation": [tf.nn.relu],
            "kernel_initializer": [kernel_initializer],
            "bias_initializer": [bias_initializer]
        })
    ])
    super(NIPSDQNCore, self).__init__(layers, kernel_initializer,
                                      bias_initializer)


class NatureDQNCore(PolicyCore):
  def __init__(self, kernel_initializer=torch_default_initializer(),
               bias_initializer=torch_default_initializer()):
    layers = layer_sequence([
        (tf.layers.Conv2D, {
          "filters": [32, 64, 64],
          "kernel_size": [8, 4, 3],
          "strides": [4, 2, 1],
          "activation": [tf.nn.relu for _ in range(3)],
          "kernel_initializer": [kernel_initializer for _ in range(3)],
          "bias_initializer": [bias_initializer for _ in range(3)]
        }),
        (tf.layers.Flatten, {}),
        (tf.layers.Dense, {
            "units": [512],
            "activation": [tf.nn.relu],
            "kernel_initializer": [kernel_initializer],
            "bias_initializer": [bias_initializer]
        })
    ])
    super(NatureDQNCore, self).__init__(layers, kernel_initializer,
                                        bias_initializer)


class UniverseStarterCore(PolicyCore):
  def __init__(self, recurrent=True):
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
    if recurrent:
      layers.append(rnn.BasicLSTMCell(num_units=256))
    else:
      layers.append(tf.layers.Dense(units=256))
    super(UniverseStarterCore, self).__init__(layers)
