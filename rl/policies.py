import abc
import math

import gym.spaces as spaces
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import rl.utils.tf_utils as tfu


USE_DEFAULT = object()

def _check_space_type(space_name, space, expected_type):
  if not isinstance(space, expected_type):
    raise ValueError(
        "{} must be an instance of gym.spaces.{}"\
            .format(space_name, expected_type.__name__))


class BaseDistributionCreator(abc.ABC):
  @abc.abstractmethod
  def create_distribution(self, tensor, action_space):
    ...

class DefaultDistributionCreator(BaseDistributionCreator):
  def create_distribution(self, tensor, action_space):
    if isinstance(action_space, spaces.Discrete):
      logits = tf.layers.dense(tensor, units=action_space.n)
      distribution = tf.distributions.Categorical(logits=logits)
      return distribution
    if isinstance(action_space, spaces.Box):
      if len(action_space.shape) > 1:
        raise TypeError(
            "action_space of type spaces.Box supported only when it has"
            " single dimension, action_space: ".format(action_space)
        )
      loc = tf.layers.dense(tensor, units=np.prod(action_space.shape))
      scale_diag = tf.get_variable(
          name="logstd",
          shape=[1, np.prod(action_space.shape)],
          initializer=tf.constant_initializer(math.log(math.e - 1))
      )
      distribution = (
          tf.contrib.distributions.MultivariateNormalDiagWithSoftplusScale(
              loc=loc, scale_diag=scale_diag)
      )
      return distribution
    raise NotImplementedError(
        "Unsupported action_space: `{}`".format(action_space))


class ValueFunctionPolicy(tfu.NetworkStructure):
  @abc.abstractmethod
  def __init__(self, name=None):
    self._inputs = None
    self._distribution = None
    self._sample = None
    self._value_preds = None

  @property
  def inputs(self):
    return self._inputs

  @property
  def state_in(self):
    return None

  @property
  def state(self):
    return None

  @property
  def distribution(self):
    return self._distribution

  @property
  def value_preds(self):
    return self._value_preds

  def act(self, observation, sess=None):
    sess = sess or tf.get_default_session()
    actions, value_preds = sess.run(
        [self._sample, self.value_preds],
        {self.inputs: observation[None, :]})
    return actions[0], value_preds[0, 0]

  def reset(self):
    pass

  def var_list(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             scope=self.scope.name)

  def preprocess_gradients(self, grad_list):
    return grad_list


class MLPPolicy(ValueFunctionPolicy):
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

  def _build(self):
    obs_shape = list(self._observation_space.shape)
    self._inputs = tf.placeholder(tf.float32, [None] + obs_shape,
                                  name="observations")
    self._build_network(self._inputs)

  def _build_network(self, inputs):
    pi = vf = inputs
    kernel_initializer = lambda scale: tfu.orthogonal_initializer(scale)
    bias_initializer = tf.zeros_initializer()
    for i in range(self._num_layers - 1):
      if self._joint:
        pi = vf = tf.layers.dense(
            pi,
            units=self._units,
            activation=tf.nn.tanh,
            kernel_initializer=kernel_initializer(np.sqrt(2)),
            bias_initializer=bias_initializer,
        )
      else:
        pi = tf.layers.dense(
            pi,
            units=self._units,
            activation=tf.nn.tanh,
            kernel_initializer=kernel_initializer(np.sqrt(2)),
            bias_initializer=bias_initializer
        )
        vf = tf.layers.dense(
            vf,
            units=self._units,
            activation=tf.nn.tanh,
            kernel_initializer=kernel_initializer(np.sqrt(2)),
            bias_initializer=bias_initializer
        )
    # _create_distribution will add last hidden layer.
    self._distribution = self._distribution_creator.create_distribution(
        pi, self._action_space)
    self._sample = self._distribution.sample()
    self._value_preds = tf.layers.dense(vf, units=1)

  def preprocess_gradients(self, grad_list):
    if self._clipping_param is not None:
      return tf.clip_by_global_norm(grad_list, self._clipping_param)[0]
    return grad_list


# TODO: CNNPolicy should be an abstract base class. What is
# actually written here is a A3C (Mnih'16) policy. It should
# be implemented separately and named accordingly.
class CNNPolicy(ValueFunctionPolicy):
  def __init__(self, observation_space, action_space, name=None):
    _check_space_type("observation_space", observation_space, spaces.Box)
    _check_space_type("action_space", action_space, spaces.Discrete)
    super(CNNPolicy, self).__init__(name=name)
    self._observation_space = observation_space
    self._action_space = action_space

  def _build(self):
    obs_shape = list(self._observation_space.shape)
    self._inputs = x = tf.placeholder(tf.float32, [None] + obs_shape,
                                      name="observations")
    self._build_network(self._inputs, self._action_space.n)

  def _build_network(self, inputs, num_actions):
    x = tf.layers.conv2d(inputs,
                         filters=16,
                         kernel_size=8,
                         strides=4,
                         activation=tf.nn.relu)
    x = tf.layers.conv2d(x,
                         filters=32,
                         kernel_size=4,
                         strides=2,
                         activation=tf.nn.relu)
    x = tf.layers.dense(x,
                        units=256,
                        activation=tf.nn.relu)
    x = tfu.flatten(x)
    self._logits = tf.layers.dense(x, units=num_actions, name="logits")
    self._distribution = tf.distributions.Categorical(logits=self._logits)
    self._sample = self._distribution.sample()
    self._value_preds = tf.layers.dense(x, units=1, name="value_preds")


class UniverseStarterPolicy(CNNPolicy):
  def __init__(self, observation_space, action_space,
               recurrent=True, name=None):
    super(UniverseStarterPolicy, self).__init__(observation_space,
                                                action_space, name=name)
    self._recurrent = recurrent
    self._state = None
    self._initial_state = None

  def _recurrent_layer(self, x):
    x = tf.expand_dims(x, [0])
    step_size = tf.shape(x)[1:2]
    lstm = rnn.BasicLSTMCell(256, state_is_tuple=True)
    self._state_in = rnn.LSTMStateTuple(
        c=tf.placeholder(tf.float32, [1, lstm.state_size.c]),
        h=tf.placeholder(tf.float32, [1, lstm.state_size.h]))
    self._initial_state = rnn.LSTMStateTuple(
        c=np.zeros(self._state_in.c.shape.as_list()),
        h=np.zeros(self._state_in.c.shape.as_list())
      )
    self._state = self._initial_state
    lstm_outputs, self._state_out = tf.nn.dynamic_rnn(
        lstm, x, initial_state=self._state_in, sequence_length=step_size,
        time_major=False)
    x = tf.reshape(lstm_outputs, [-1, lstm.output_size])
    return x

  def _dense_layer(self, x):
    return tf.layers.dense(x, units=256, activation=tf.nn.elu)

  def _build_network(self, inputs, num_actions):
    x = inputs
    for i in range(4):
      x = tf.layers.conv2d(x,
                           filters=32,
                           kernel_size=3,
                           strides=2,
                           padding="same",
                           activation=tf.nn.elu,
                           name="conv2d_{}".format(i + 1))
    x = tfu.flatten(x)
    if self._recurrent:
      x = self._recurrent_layer(x)
    else:
      x = self._dense_layer(x)
    self._logits = tf.layers.dense(
        x,
        units=num_actions,
        kernel_initializer=tfu.normalized_columns_initializer(0.01),
        name="logits")
    self._distribution = tf.distributions.Categorical(logits=self._logits)
    self._sample = self._distribution.sample()
    self._value_preds = tf.layers.dense(
        x,
        units=1,
        kernel_initializer=tfu.normalized_columns_initializer(1.0),
        name="value_preds")

  @property
  def state_in(self):
    """ Tensorflow placeholder for state `LSTMStateCell` input. """
    return self._state_in

  @property
  def state(self):
    return self._state

  def act(self, observation, sess=None):
    sess = sess or tf.get_default_session()
    fetches = [self._sample, self.value_preds]
    feed_dict = {self.inputs: observation[None, :]}
    if self._recurrent:
      fetches.append(self._state_out)
      feed_dict[self._state_in] = self._state
      actions, value_preds, self._state = sess.run(fetches, feed_dict)
    else:
      actions, value_preds = sess.run(fetches, feed_dict)
    return actions[0], value_preds[0, 0]

  def reset(self):
    self._state = self._initial_state

  def preprocess_gradients(self, grad_list):
    if self._recurrent:
      return tf.clip_by_global_norm(grad_list, 40.0)[0]
    else:
      return grad_list
