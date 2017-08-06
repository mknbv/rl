import functools
import numpy as np
import tensorflow as tf

def lazy_function(function):
    attribute = '_cache_' + function.__name__
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

def flatten(t):
  return tf.reshape(t, [-1, np.prod(t.shape[1:]).value])

def normalized_columns_initializer(stddev=1.0):
  def _initializer(shape, dtype=None, partition_info=None):
    out = np.random.randn(*shape).astype(np.float32)
    out *= stddev / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)
  return _initializer

def partial_restore(variables, checkpoint, session=None, scope=None):
  meta_file = checkpoint + ".meta"
  with tf.Graph().as_default(), tf.Session() as sess, sess.as_default():
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, checkpoint)
    checkpoint_vars = {
        v.name: v.eval()
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope=scope)
    }
  for v in variables:
    v.load(checkpoint_vars[v.name])
