import abc

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from .distribution import DefaultDistributionCreator
from .core import BasePolicy, MLPCore, UniverseStarterCore
import rl.utils.tf_utils as tfu
from rl.utils.env_batch import SpaceBatch


USE_DEFAULT = object()


class ActorCriticPolicy(BasePolicy):
  @abc.abstractmethod
  def __init__(self, name=None):
    super(ActorCriticPolicy, self).__init__(name=name)
    self._distribution = None
    self._sample = None
    self._log_prob = None
    self._critic_tensor = None

  @property
  def distribution(self):
    return self._distribution

  @property
  def critic_tensor(self):
    return self._critic_tensor

  def act(self, observations, sess=None):
    sess = sess or tf.get_default_session()
    fetches = {
        "actions": self._sample,
        "critic_values": self.critic_tensor
    }
    if self._log_prob is not None:
      fetches["log_probs"] = self._log_prob
    return sess.run(fetches, {self.observations: observations})


class MLPPolicy(ActorCriticPolicy):
  def __init__(self,
               observation_space,
               action_space,
               distribution_creator=USE_DEFAULT,
               num_layers=3,
               units=64,
               clipping_param=10,
               compute_log_prob=False,
               joint=False,
               name=None):
    super(MLPPolicy, self).__init__(name=name)
    self._observation_space = observation_space
    self._action_space = action_space
    if distribution_creator == USE_DEFAULT:
      distribution_creator = DefaultDistributionCreator()
    self._distribution_creator = distribution_creator
    self._num_layers = num_layers
    self._units = units
    self._clipping_param = clipping_param
    self._compute_log_prob = compute_log_prob
    self._joint = joint
    # _create_distribution will add the last hidden layer.
    self._policy_core = MLPCore(num_layers=num_layers-1, units=units)
    if self._joint:
      self._value_core = None
    else:
      self._value_core = MLPCore(num_layers=num_layers-1, units=units)

  def _build(self):
    obs_shape = list(self._observation_space.shape)
    self._observations = tf.placeholder(tf.float32, [None] + obs_shape,
                                        name="observations")
    self._build_network(self._observations)

  def _build_network(self, inputs):
    x = pi = critic = inputs
    if self._joint:
      for layer in self._policy_core.layers:
        x = pi = critic = layer.apply(x)
    else:
      for pi_layer, critic_layer in zip(self._policy_core.layers,
                                        self._value_core.layers):
        pi = pi_layer.apply(pi)
        critic = critic_layer.apply(critic)
    self._distribution = self._distribution_creator.create_distribution(
        pi, self._action_space)
    self._sample = self._distribution.sample()
    if self._compute_log_prob:
      self._log_prob = self.distribution.log_prob(self._sample)
    self._critic_tensor = tf.layers.dense(
      critic,
      units=1,
      kernel_initializer=self._value_core.kernel_initializer,
      bias_initializer=self._value_core.bias_initializer,
      name="critic"
    )

  def preprocess_gradients(self, grad_list):
    if self._clipping_param is not None:
      return tf.clip_by_global_norm(grad_list, self._clipping_param)[0]
    return grad_list


class CategoricalActorCriticPolicy(ActorCriticPolicy):
  def __init__(self, observation_space, action_space, core,
               ubyte_rescale=True,
               compute_log_prob=False,
               name=None):
    super(CategoricalActorCriticPolicy, self).__init__(name=name)
    self._observation_space = observation_space
    self._action_space = action_space
    self._core = core
    self._compute_log_prob = compute_log_prob
    self._ubyte_rescale = ubyte_rescale

  def _build(self):
    obs_type = self._observation_space.dtype
    obs_shape = (None,) + self._observation_space.shape
    self._observations = tf.placeholder(obs_type, obs_shape,
                                        name="observations")

    x = tf.to_float(self._observations)
    if obs_type == np.uint8 and self._ubyte_rescale:
      x = x / 255.0
    for layer in self._core.layers:
      x = layer.apply(x)

    self._logits = tf.layers.dense(
        x,
        units=self._action_space.n,
        kernel_initializer=self._core.kernel_initializer,
        bias_initializer=self._core.bias_initializer,
        name="logits")
    self._distribution = tf.distributions.Categorical(logits=self._logits)
    self._sample = self._distribution.sample()
    if self._compute_log_prob:
      self._log_prob = self.distribution.log_prob(self._sample)
    self._critic_tensor = tf.layers.dense(
        x,
        units=1,
        kernel_initializer=self._core.kernel_initializer,
        bias_initializer=self._core.bias_initializer,
        name="critic")


class UniverseStarterPolicy(ActorCriticPolicy):
  def __init__(self, observation_space, action_space,
               recurrent=True, compute_log_prob=False, name=None):
    super(UniverseStarterPolicy, self).__init__(name=name)
    self._state_inputs = None
    self._state_values = None
    self._state_outputs = None
    self._observation_space = observation_space
    self._action_space = action_space
    self._recurrent = recurrent
    self._compute_log_prob = compute_log_prob
    self._core = UniverseStarterCore(recurrent=recurrent)

  def _build(self):
    obs_shape = list(self._observation_space.shape)
    self._observations = tf.placeholder(tf.float32, [None] + obs_shape,
                                        name="observations")
    x = self._observations
    for layer in self._core.layers:
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
    if self._compute_log_prob:
      self._log_prob = self._distribution.log_prob(self._sample)
    self._critic_tensor = tf.layers.dense(
        x,
        units=1,
        kernel_initializer=tfu.normalized_columns_initializer(1.0),
        name="critic"
    )

  def _apply_recurrent_layer(self, x, layer):
    self._state_inputs = rnn.LSTMStateTuple(
        c=tf.placeholder(tf.float32, [None, layer.state_size.c],
                         name="lstm_c"),
        h=tf.placeholder(tf.float32, [None, layer.state_size.h],
                         name="lstm_h"))
    batch_size = tf.shape(self._state_inputs.c)[0]
    tf.assert_equal(tf.shape(self._state_inputs.h)[0], batch_size)

    x = tf.reshape(x, (batch_size, -1) + tuple(x.shape[1:]))
    step_size = tf.shape(x)[1]

    if isinstance(self._observation_space, SpaceBatch):
      num_envs = len(self._observation_space.spaces)
    else:
      num_envs = 1
    self._initial_state_values = rnn.LSTMStateTuple(
        c=np.zeros([num_envs, layer.state_size.c],
                   dtype=np.float32),
        h=np.zeros([num_envs, layer.state_size.h],
                   dtype=np.float32))
    self._state_values = self._initial_state_values

    sequence_length = tf.fill([batch_size], step_size)
    layer_outputs, self._state_outputs = tf.nn.dynamic_rnn(
        layer, x, initial_state=self._state_inputs,
        sequence_length=sequence_length,
        time_major=False)
    x = tf.reshape(layer_outputs, [-1, layer.output_size])
    return x

  @property
  def state_inputs(self):
    return self._state_inputs

  @property
  def state_values(self):
    return self._state_values

  def reset(self, mask):
    if self._recurrent:
      self._state_values = rnn.LSTMStateTuple(
          c=np.where(mask[:, None],
                     self._initial_state_values.c,
                     self.state_values.c),
          h=np.where(mask[:, None],
                     self._initial_state_values.h,
                     self.state_values.h))

  def act(self, observations, sess=None):
    sess = sess or tf.get_default_session()
    fetches = {"actions": self._sample, "critic_values": self._critic_tensor}
    if self._log_prob is not None:
      fetches["log_probs"] = self._log_prob
    feed_dict = {self.observations: observations}
    if self._recurrent:
      fetches["state_values"] = self._state_outputs
      feed_dict[self.state_inputs] = self.state_values
      values = sess.run(fetches, feed_dict)
      self._state_values = values["state_values"]
      values.pop("state_values")
    else:
      values = sess.run(fetches, feed_dict)
    return values

  def preprocess_gradients(self, grad_list):
    if self._recurrent:
      return tf.clip_by_global_norm(grad_list, 40.0)[0]
    else:
      return grad_list
