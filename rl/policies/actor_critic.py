import abc
import math

import gym.spaces as spaces
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from .distribution import DefaultDistributionCreator
from .core import (_check_space_type, BasePolicy, MLPCore, DQNCore,
                   UniverseStarterCore)
import rl.utils.tf_utils as tfu


__all__ = [
    "ActorCriticPolicy",
    "MLPPolicy",
    "A3CAtariPolicy",
    "UniverseStarterPolicy"
]

USE_DEFAULT = object()


class ActorCriticPolicy(BasePolicy):
  @abc.abstractmethod
  def __init__(self, name=None):
    super(ActorCriticPolicy, self).__init__(name=name)
    self._distribution = None
    self._sample = None
    self._critic_tensor = None

  @property
  def distribution(self):
    return self._distribution

  @property
  def critic_tensor(self):
    return self._critic_tensor

  def act(self, observation, sess=None):
    sess = sess or tf.get_default_session()
    actions, critic_values = sess.run(
        [self._sample, self.critic_tensor],
        {self.observations: observation[None, :]})
    return actions[0], critic_values[0, 0]


class MLPPolicy(ActorCriticPolicy):
  def __init__(self,
               observation_space,
               action_space,
               distribution_creator=USE_DEFAULT,
               num_layers=3,
               units=64,
               clipping_param=10,
               joint=False,
               name=None):
    _check_space_type("observation_space", observation_space, spaces.Box)
    super(MLPPolicy, self).__init__(name=name)
    self._observation_space = observation_space
    self._action_space = action_space
    if distribution_creator == USE_DEFAULT:
      distribution_creator = DefaultDistributionCreator()
    self._distribution_creator = distribution_creator
    self._num_layers = num_layers
    self._units = units
    self._clipping_param = clipping_param
    self._joint = joint
    # _create_distribution will add the last hidden layer.
    if self._joint:
      self._core_layers = mlp_core(num_layers=num_layers-1, units=units)
    else:
      self._core_layers = [
          mlp_core(num_layers=num_layers-1, units=units),
          mlp_core(num_layers=num_layers-1, units=units)
      ]

  def _build(self):
    obs_shape = list(self._observation_space.shape)
    self._observations = tf.placeholder(tf.float32, [None] + obs_shape,
                                        name="observations")
    self._build_network(self._observations)

  def _build_network(self, inputs):
    x = pi = critic = inputs
    if self._joint:
      for layer in self._core_layers:
        x = pi = critic = layer.apply(x)
    else:
      for pi_layer, critic_layer in zip(*self._core_layers):
        pi = pi_layer.apply(pi)
        critic = critic_layer.apply(critic)
    self._distribution = self._distribution_creator.create_distribution(
        pi, self._action_space)
    self._sample = self._distribution.sample()
    self._critic_tensor = tf.layers.dense(critic, units=1, name="critic")

  def preprocess_gradients(self, grad_list):
    if self._clipping_param is not None:
      return tf.clip_by_global_norm(grad_list, self._clipping_param)[0]
    return grad_list


class A3CAtariPolicy(ActorCriticPolicy):
  metadata = {"visualize_observations": True}

  def __init__(self, observation_space, action_space, name=None):
    _check_space_type("observation_space", observation_space, spaces.Box)
    _check_space_type("ation_space", action_space, spaces.Discrete)
    super(A3CAtariPolicy, self).__init__(name=name)
    self._observation_space = observation_space
    self._action_space = action_space
    self._core_layers = dqn_core(kind="nips")

  def _build(self):
    self._observations = tf.placeholder(tf.float32, [None] + obs_shape,
                                        name="observations")

    x = self._observations
    for layer in self._core_layers:
      x = layer.apply(x)

    self._logits = tf.layers.dense(x, units=self._action_space.n, name="logits")
    self._distribution = tf.distributions.Categorical(logits=self._logits)
    self._sample = self._distribution.sample()
    self._critic_tensor = tf.layers.dense(x, units=1, name="critic")


class UniverseStarterPolicy(ActorCriticPolicy):
  metadata = {"visualize_observations": True}

  def __init__(self, observation_space, action_space,
               recurrent=True, name=None):
    _check_space_type("observation_space", observation_space, spaces.Box)
    _check_space_type("action_space", action_space, spaces.Discrete)
    super(UniverseStarterPolicy, self).__init__(name=name)
    self._state_inputs = None
    self._state_values = None
    self._state_outputs = None
    self._observation_space = observation_space
    self._action_space = action_space
    self._recurrent = recurrent
    self._core = UniverseStarterCore(recurrent=recurrent)

  def _build(self):
    obs_shape = list(self._observation_space.shape)
    self._observations = tf.placeholder(tf.float32, [None] + obs_shape,
                                        name="observations")
    x = self._observations
    for layer in self._core():
      if isinstance(layer, rnn.RNNCell):
        x = self._apply_recurrent_layer(x, layer)
      else:
        x = layer.apply(x)

    self._logits = tf.layers.dense(
        x,
        units=self._action_space.n,
        kernel_initializer=tfu.normalized_columns_initializer(0.01),
        name="logits"
    )
    self._distribution = tf.distributions.Categorical(logits=self._logits)
    self._sample = self._distribution.sample()
    self._critic_tensor = tf.layers.dense(
        x,
        units=1,
        kernel_initializer=tfu.normalized_columns_initializer(1.0),
        name="critic"
    )

  def _apply_recurrent_layer(self, x, layer):
    x = tf.expand_dims(x, [0])
    step_size = tf.shape(x)[1:2]
    self._state_inputs = rnn.LSTMStateTuple(
        c=tf.placeholder(tf.float32, [1, layer.state_size.c]),
        h=tf.placeholder(tf.float32, [1, layer.state_size.h]))
    self._initial_state_values = rnn.LSTMStateTuple(
        c=np.zeros(self._state_inputs.c.shape.as_list()),
        h=np.zeros(self._state_inputs.c.shape.as_list())
      )
    self._state_values = self._initial_state_values
    layer_outputs, self._state_outputs = tf.nn.dynamic_rnn(
        layer, x, initial_state=self._state_inputs, sequence_length=step_size,
        time_major=False)
    x = tf.reshape(layer_outputs, [-1, layer.output_size])
    return x

  @property
  def state_inputs(self):
    return self._state_inputs

  @property
  def state_values(self):
    return self._state_values

  def reset(self):
    self._state_values = self._initial_state_values

  def act(self, observation, sess=None):
    sess = sess or tf.get_default_session()
    fetches = [self._sample, self.critic_tensor]
    feed_dict = {self.observations: observation[None, :]}
    if self._recurrent:
      fetches.append(self._state_outputs)
      feed_dict[self.state_inputs] = self.state_values
      actions, critic_values, self._state_values = sess.run(fetches, feed_dict)
    else:
      actions, critic_values = sess.run(fetches, feed_dict)
    return actions[0], critic_values[0, 0]

  def preprocess_gradients(self, grad_list):
    if self._recurrent:
      return tf.clip_by_global_norm(grad_list, 40.0)[0]
    else:
      return grad_list
