import abc

from gym import spaces
import tensorflow as tf


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
