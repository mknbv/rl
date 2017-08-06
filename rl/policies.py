import gym.spaces as spaces
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from rl.distributions import Categorical
import rl.tf_utils as U


def check_space_type(space_name, space, expected_type):
  if not isinstance(space, expected_type):
    raise ValueError(
        "{} must be an instance of gym.spaces.{}"\
            .format(space_name, expected_type.__name__))


class ValueFunctionPolicy(object):
  def __init__(self):
    self.scope = None
    self.inputs = None
    self.distribution = None
    self.value_preds = None

  def act(self, observation, sess=None):
    if sess is None:
      sess = tf.get_default_session()
    actions, value_preds = sess.run(
        [self.distribution.sample(), self.value_preds],
        {self.inputs: observation[None, :]})
    return actions[0], value_preds[0, 0]

  def reset(self):
    pass

  def var_list(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                             scope=self.scope.name)

  def gradient_preprocessing(self, grad_list):
    return grad_list

class SimplePolicy(ValueFunctionPolicy):
  def __init__(self, observation_space, action_space, name=None):
    check_space_type("observation_space", observation_space, spaces.Box)
    check_space_type("action_space", action_space, spaces.Discrete)
    if name is None:
      name = self.__class__.__name__
    with tf.variable_scope(None, name) as scope:
      self.scope = scope
      obs_shape = list(observation_space.shape)
      self.inputs = x = tf.placeholder(tf.float32, [None] + obs_shape,
                                       name="observations")
      self._init_network(self.inputs, action_space.n)

  def _init_network(self, inputs, num_actions):
    x = tf.layers.dense(inputs, units=16, activation=tf.nn.tanh)
    self.logits = tf.layers.dense(x, units=num_actions)
    self.distribution = Categorical.from_logits(self.logits)
    self.value_preds = tf.layers.dense(x, units=1)


class CNNPolicy(ValueFunctionPolicy):
  def __init__(self, observation_space, action_space,
               name=None):
    check_space_type("observation_space", observation_space, spaces.Box)
    check_space_type("action_space", action_space, spaces.Discrete)
    if name is None:
      name = self.__class__.__name__
    with tf.variable_scope(None, name) as scope:
      self.scope = scope
      obs_shape = list(observation_space.shape)
      self.inputs = tf.placeholder(tf.float32, [None] + obs_shape,
                                   name="observations")
      self._init_network(self.inputs, action_space.n)

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
    self.logits = tf.layers.dense(x, units=num_actions, name="logits")
    self.distribution = Categorical.from_logits(self.logits)
    self.value_preds = tf.layers.dense(x, units=1, name="value_preds")


class UniverseStarterPolicy(CNNPolicy):
  def _init_network(self, inputs, num_actions):
    x = inputs
    for i in range(4):
      x = tf.layers.conv2d(x,
                           filters=32,
                           kernel_size=3,
                           strides=2,
                           activation=tf.nn.elu,
                           name="conv2d_{}".format(i + 1))
    x = tf.expand_dims(U.flatten(x), [0])
    step_size = tf.shape(self.inputs)[:1]
    lstm = rnn.BasicLSTMCell(256, state_is_tuple=True)
    self.state_in = rnn.LSTMStateTuple(
        c=tf.zeros([1, lstm.state_size.c], tf.float32),
        h=tf.zeros([1, lstm.state_size.h], tf.float32))
    self.prev_state = None
    lstm_outputs, self.state_out = tf.nn.dynamic_rnn(
        lstm, x, initial_state=self.state_in, sequence_length=step_size,
        time_major=False)
    x = tf.reshape(lstm_outputs, [-1, lstm.output_size])
    self.logits = tf.layers.dense(
        x,
        units=num_actions,
        kernel_initializer=U.normalized_columns_initializer(0.01),
        name="logits")
    self.distribution = Categorical.from_logits(self.logits)
    self.value_preds = tf.layers.dense(
        x,
        units=1,
        kernel_initializer=U.normalized_columns_initializer(1.0),
        name="value_preds")

  def act(self, observation, sess=None):
    if sess is None:
      sess = tf.get_default_session()
    fetches = [self.distribution.sample(), self.value_preds, self.state_out]
    feed_dict = {self.inputs: observation[None, :]}
    if self.prev_state is not None:
      feed_dict[self.state_in] = self.prev_state
    actions, value_preds, self.prev_state = sess.run(fetches, feed_dict)
    return actions[0], value_preds[0]

  def reset(self):
    self.prev_state = None

  def gradient_preprocessing(self, grad_list):
    return tf.clip_by_global_norm(grad_list, 40.0)[0]
