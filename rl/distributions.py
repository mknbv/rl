import tensorflow as tf

from rl.tf_utils import lazy_function


class BaseDistribution(object):
  def __init__(self):
    pass

  @property
  def dtype(self):
    return self.sample().dtype

  @property
  def shape(self):
    return self.sample().shape

  def sample(self):
    raise NotImplementedError()

  def neglogp(self, x):
    raise NotImplementedError()

  def logp(self, x):
    return -self.neglogp(x)

  def entropy(self, other):
    raise NotImplementedError()


class Categorical(BaseDistribution):
  def __init__(self, logits):
    self.name = "categorical_distribution"
    with tf.variable_scope(self.name):
      tf.assert_rank(logits, 2)
      self.logits = tf.cast(logits, tf.float32)

  @classmethod
  def from_logits(cls, logits):
    return cls(logits)

  @lazy_function
  def sample(self):
    with tf.variable_scope(self.name, reuse=True):
      # https://github.com/tensorflow/tensorflow/issues/2774
      # seems to be fixed in tensorflow 1.3
      logits = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
      return tf.reshape(tf.multinomial(logits, num_samples=1), [-1])

  def neglogp(self, x):
    with tf.variable_scope(self.name, reuse=True):
      return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                            labels=x)

  @lazy_function
  def entropy(self):
    with tf.variable_scope(self.name, reuse=True):
      probas = tf.nn.softmax(self.logits)
      logp = tf.nn.log_softmax(self.logits)
      return -tf.reduce_sum(probas * logp, axis=-1)


class Gaussian(BaseDistribution):
  def __init__(self, mean, logstdev):
    self.name = "gaussian_distribution"
    with tf.variable_scope(self.name):
      tf.assert_rank(mean, 2)
      tf.assert_rank(logstdev, 2)
      self.mean = tf.cast(mean, tf.float32)
      self.logstdev = tf.cast(logstdev, tf.float32)

  @lazy_function
  def sample(self):
    with tf.variable_scope(self.name, reuse=True):
      z = tf.random_normal(tf.shape(self.mean))
      return self.mean + tf.exp(self.logstdev) ** (-0.5) * z

  def neglogp(self, x):
    with tf.variable_scope(self.name, reuse=True):
      t1 = tf.reduce_sum(self.logstdev, axis=-1, keep_dims=True)
      t2 = tf.shape(self.mean)[1] * tf.log(2 * np.pi)
      t3 = tf.reduce_sum(
          (x - self.mean) ** 2 * tf.exp(self.logstdev) ** (-0.5),
          axis=-1, keep_dims=True)
      return -0.5 * (t1 + t2 + t3)

  @lazy_function
  def entropy(self):
    with tf.variable_scope(self.name, reuse=True):
      raise NotImplementedError()
