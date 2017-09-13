import abc
import gym.spaces as spaces
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from rl.distributions import Categorical
import rl.tf_utils as U


def _check_space_type(space_name, space, expected_type):
  if not isinstance(space, expected_type):
    raise ValueError(
        "{} must be an instance of gym.spaces.{}"\
            .format(space_name, expected_type.__name__))


class ValueFunctionPolicy(abc.ABC):
  @abc.abstractmethod
  def __init__(self, observation_space, action_space, name):
    self._scope = None
    self._inputs = None
    self._distribution = None
    self._value_preds = None

  def act(self, observation, sess=None):
    sess = sess or tf.get_default_session()
    actions, value_preds = sess.run(
        [self.distribution.sample(), self.value_preds],
        {self.inputs: observation[None, :]})
    return actions[0], value_preds[0, 0]

  def reset(self):
    pass

  def var_list(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             scope=self._scope.name)

  def preprocess_gradients(self, grad_list):
    return grad_list

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



class SimplePolicy(ValueFunctionPolicy):
  def __init__(self, observation_space, action_space, name=None):
    _check_space_type("observation_space", observation_space, spaces.Box)
    _check_space_type("action_space", action_space, spaces.Discrete)
    if name is None:
      name = self.__class__.__name__
    with tf.variable_scope(None, name) as scope:
      self._scope = scope
      obs_shape = list(observation_space.shape)
      self._inputs = x = tf.placeholder(tf.float32, [None] + obs_shape,
                                       name="observations")
      self._init_network(self._inputs, action_space.n)

  def _init_network(self, inputs, num_actions):
    x = tf.layers.dense(inputs, units=16, activation=tf.nn.tanh)
    self._logits = tf.layers.dense(x, units=num_actions)
    self._distribution = Categorical.from_logits(self._logits)
    self._value_preds = tf.layers.dense(x, units=1)


class CNNPolicy(ValueFunctionPolicy):
  def __init__(self, observation_space, action_space,
               name=None):
    _check_space_type("observation_space", observation_space, spaces.Box)
    _check_space_type("action_space", action_space, spaces.Discrete)
    if name is None:
      name = self.__class__.__name__
    with tf.variable_scope(None, name) as scope:
      self._scope = scope
      obs_shape = list(observation_space.shape)
      self._inputs = tf.placeholder(tf.float32, [None] + obs_shape,
                                   name="observations")
      self._init_network(self._inputs, action_space.n)

  def _init_network(self, inputs, num_actions):
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
    x = U.flatten(x)
    self._logits = tf.layers.dense(x, units=num_actions, name="logits")
    self._distribution = Categorical.from_logits(self._logits)
    self._value_preds = tf.layers.dense(x, units=1, name="value_preds")


class UniverseStarterPolicy(CNNPolicy):
  def _init_network(self, inputs, num_actions):
    x = inputs
    for i in range(4):
      x = tf.layers.conv2d(x,
                           filters=32,
                           kernel_size=3,
                           strides=2,
                           padding="same",
                           activation=tf.nn.elu,
                           name="conv2d_{}".format(i + 1))
    x = tf.expand_dims(U.flatten(x), [0])
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
        kernel_initializer=U.normalized_columns_initializer(0.01),
        name="logits")
    self._distribution = Categorical.from_logits(self._logits)
    self._value_preds = tf.layers.dense(
        x,
        units=1,
        kernel_initializer=U.normalized_columns_initializer(1.0),
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
        [self.distribution.sample(), self.value_preds, self._state_out],
        {self.inputs: observation[None, :], self._state_in: self._state})
    return actions[0], value_preds[0, 0]

  def reset(self):
    self._state = self._initial_state

  def preprocess_gradients(self, grad_list):
    return tf.clip_by_global_norm(grad_list, 40.0)[0]
