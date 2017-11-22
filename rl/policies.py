import abc
import gym.spaces as spaces
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

import rl.utils.tf_utils as tfu


def _check_space_type(space_name, space, expected_type):
  if not isinstance(space, expected_type):
    raise ValueError(
        "{} must be an instance of gym.spaces.{}"\
            .format(space_name, expected_type.__name__))


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


class SimplePolicy(ValueFunctionPolicy):
  def __init__(self, observation_space, action_space, name=None):
    _check_space_type("observation_space", observation_space, spaces.Box)
    _check_space_type("action_space", action_space, spaces.Discrete)
    super(SimplePolicy, self).__init__(name=name)
    self._observation_space = observation_space
    self._action_space = action_space

  def _build(self):
    obs_shape = list(self._observation_space.shape)
    self._inputs = x = tf.placeholder(tf.float32, [None] + obs_shape,
                                      name="observations")
    self._build_network(self._inputs, self._action_space.n)

  def _build_network(self, inputs, num_actions):
    x = tf.layers.dense(inputs, units=16, activation=tf.nn.tanh)
    self._logits = tf.layers.dense(x, units=num_actions)
    self._distribution = tf.distributions.Categorical(logits=self._logits)
    self._sample = self._distribution.sample()
    self._value_preds = tf.layers.dense(x, units=1)


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
    x = tf.expand_dims(tfu.flatten(x), [0])
    step_size = tf.shape(inputs)[:1]
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
    actions, value_preds, self._state = sess.run(
        [self._sample, self.value_preds, self._state_out],
        {self.inputs: observation[None, :], self._state_in: self._state})
    return actions[0], value_preds[0, 0]

  def reset(self):
    self._state = self._initial_state

  def preprocess_gradients(self, grad_list):
    return tf.clip_by_global_norm(grad_list, 40.0)[0]
