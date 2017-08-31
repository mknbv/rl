from contextlib import contextmanager
import tensorflow as tf

USE_DEFAULT = object()


class SingularTrainingManager(object):
  def __init__(self,
               logdir,
               summary_period,
               checkpoint_period,
               saver=None,
               checkpoint=None,
               config=USE_DEFAULT):
    self.logdir = logdir
    self.summary_period = summary_period
    self.saver = saver or tf.train.Saver()
    self.hooks = [
        tf.train.CheckpointSaverHook(
          logdir, saver=saver, save_steps=checkpoint_period)
      ]
    self.checkpoint = checkpoint
    if config == USE_DEFAULT:
      config = self._get_default_config()
    self.config = config

  def _get_default_config(self):
    return tf.ConfigProto(intra_op_parallelism_threads=1,
                          inter_op_parallelism_threads=2)

  @contextmanager
  def session(self):
    with tf.train.SingularMonitoredSession(hooks=self.hooks,
                                           config=self.config) as sess:
      if self.checkpoint is not None:
        self.saver.restore(sess, self.checkpoint)
      yield sess


class DistributedTrainingManager(SingularTrainingManager):
  def __init__(self,
               target,
               is_chief,
               logdir,
               summary_period,
               checkpoint_period,
               saver=None,
               checkpoint=None,
               config=USE_DEFAULT):
    if saver is None:
      saver = tf.train.Saver(sharded=True)
    super(DistributedTrainingManager, self).__init__(
        logdir=logdir,
        summary_period=summary_period,
        checkpoint_period=checkpoint_period,
        saver=saver,
        checkpoint=checkpoint,
        config=config)
    self.target = target
    self.is_chief = is_chief

  @contextmanager
  def session(self):
    if self.is_chief:
      session_creator = tf.train.ChiefSessionCreator(
          master=self.target, config=self.config,
          checkpoint_filename_with_path=self.checkpoint)
    else:
      session_creator = tf.train.WorkerSessionCreator(
          master=self.target, config=self.config)
    with tf.train.MonitoredSession(session_creator, hooks=self.hooks) as sess:
      yield sess
