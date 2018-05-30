from copy import deepcopy

from gym import spaces
import numpy as np
import tensorflow as tf

from .core import _check_space_type, BasePolicy


__all__ = [
    "ValueBasedPolicy",
    "DistributionalPolicy",
]


class ValueBasedPolicy(BasePolicy):
  def __init__(self, **kwargs):
    _check_space_type("action_space", kwargs["action_space"], spaces.Discrete)
    self._action_space = kwargs["action_space"]
    super(ValueBasedPolicy, self).__init__(name=kwargs["name"])
    if kwargs["is_target"] and kwargs["target"] is not None:
      raise ValueError("When is_target is True target must be None")
    self._epsilon = kwargs["epsilon"]
    self._values = None
    self._is_target = kwargs["is_target"]
    self._target = kwargs["target"]
    if not self._is_target and self._target is None:
      kwargs["name"] = self.scope.name + "_target"
      kwargs["is_target"] = True
      kwargs["core"] = deepcopy(kwargs["core"])
      self._target = self.__class__(**kwargs)

  @property
  def epsilon(self):
    return self._epsilon

  @property
  def values(self):
    return self._values

  @property
  def target(self):
    return self._target

  def act(self, observations, epsilon_value=None, sess=None):
    sess = sess or tf.get_default_session()
    if epsilon_value is None:
      epsilon_value = sess.run(self.epsilon)
    if np.random.random() <= epsilon_value:
      return {"actions": np.asarray([self._action_space.sample()
                                     for _ in range(observations.shape[0])])}
    else:
      values = sess.run(self.values, {self.observations: observations})
      return {"actions": np.argmax(values, axis=-1)}


class DistributionalPolicy(ValueBasedPolicy):
  metadata = {"visualize_observations": True}

  def __init__(self, observation_shape, observation_type, action_space,
               core, epsilon, nbins=200, ubyte_rescale=True,
               is_target=False, target=None, name=None):
    kwargs = locals()
    kwargs.pop("self")
    kwargs.pop("__class__")
    super(DistributionalPolicy, self).__init__(**kwargs)
    self._observation_shape = observation_shape
    self._observation_type = observation_type
    self._nbins = nbins
    self._ubyte_rescale = ubyte_rescale
    self._core = core
    self._output_tensor = None

    if not self._is_target:
      self.add_after_build_hook(lambda *args, **kwargs: self.target.build())

  @property
  def output_tensor(self):
    """ Either probability values in each of the bins or quantiles. """
    return self._output_tensor

  def _build(self):
    obs_shape = (None,) + self._observation_space.shape
    self._observations = tf.placeholder(self._observation_space.dtype,
                                        obs_shape, name="observations")
    x = tf.to_float(self._observations)
    if self._observations.dtype == np.uint8 and self._ubyte_rescale:
      x = x / 255.0

    for layer in self._core.layers:
      if isinstance(layer, tf.contrib.rnn.RNNCell):
        raise TypeError(
          "DistriubutionalPolicy does not support recurrent layers,"
          " found {}".format(layer)
        )
      x = layer.apply(x)

    final_units = self._action_space.n * self._nbins
    x = tf.layers.dense(
        x,
        units=final_units,
        activation=None,
        kernel_initializer=self._core.kernel_initializer,
        bias_initializer=self._core.bias_initializer,
        name="output"
    )
    self._output_tensor = tf.reshape(
        x, [-1, self._action_space.n, self._nbins])
    self._values = tf.reduce_mean(self.output_tensor, axis=-1, name="values")
