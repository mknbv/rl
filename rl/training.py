from contextlib import contextmanager
import logging

import tensorflow as tf

from rl.utils.tf_utils import purge_orphaned_summaries

USE_DEFAULT = object()


class DistributedTrainer(object):
  def __init__(self,
               target,
               is_chief,
               logdir,
               summary_period,
               checkpoint_period,
               checkpoint=None,
               config=USE_DEFAULT):
    self._target = target
    self._is_chief = is_chief
    self._logdir = logdir
    self._summary_period = summary_period
    self._checkpoint_period = checkpoint_period
    self._checkpoint = checkpoint
    if config == USE_DEFAULT:
      config = self._get_default_config()
    self._config = config
    self._session = None
    self._last_summary_step = None

  def _get_default_config(self):
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=2)
    # Dynamic memory allocation on gpu.
    # https://github.com/tensorflow/tensorflow/issues/1578
    config.gpu_options.allow_growth = True
    return config

  @property
  def summary_writer(self):
    return tf.summary.FileWriterCache.get(self._logdir)

  @contextmanager
  def managed_session(self, hooks=None):
    # Take only ps variables to save and check if initialization completed.
    ps_variables = list(filter(lambda v: v.device.startswith("/job:ps/"),
                               tf.global_variables()))
    ready_op = tf.report_uninitialized_variables(ps_variables)
    saver = tf.train.Saver(ps_variables)
    scaffold = tf.train.Scaffold(ready_for_local_init_op=ready_op,
                                 ready_op=ready_op, saver=saver)
    if self._is_chief and hooks is None:
      hooks = [
          tf.train.CheckpointSaverHook(
              self._logdir, saver=saver,
              save_steps=self._checkpoint_period
          ),
      ]
      session_creator = tf.train.ChiefSessionCreator(
          scaffold=scaffold, master=self._target,
          config=self._config, checkpoint_filename_with_path=self._checkpoint)
    else:
      session_creator = tf.train.WorkerSessionCreator(
          scaffold=scaffold, master=self._target, config=self._config)
    with tf.train.MonitoredSession(session_creator, hooks=hooks) as sess:
      self._session = sess
      yield sess

  def step(self, algorithm=None, fetches=None,
           feed_dict=None, summary_time=None, sess=None):
    if summary_time and algorithm is None:
      raise TypeError("Algorithm cannot be None when summary_time is True")

    if sess is None:
      sess = self._session or tf.get_default_session()
    global_step = tf.train.get_global_step()
    step = sess.run(global_step)

    if (summary_time is None
        and algorithm is not None
        and self._is_chief
        and (self._last_summary_step is None\
             or step - self._last_summary_step >= self._summary_period)):
        summary_time = True

    run_fetches = {}
    if fetches is not None:
      run_fetches["fetches"] = fetches
    if algorithm is not None:
      run_fetches["train_op"] = algorithm.train_op
      if summary_time:
        run_fetches["logging"] = algorithm.logging_fetches
        run_fetches["summaries"] = algorithm.summaries

    if feed_dict is None:
      feed_dict = {}
    algorithm_feed_dict = algorithm.get_feed_dict(sess,
                                                  summary_time=summary_time)
    if len(algorithm_feed_dict.keys() & feed_dict.keys()) > 0:
      intersection = algorithm_feed_dict.keys() & feed_dict.keys()
      raise ValueError(
          "Algorithm feed dict intersects with the given feed dict: {}"
          .format(intersection)
      )
    feed_dict.update(algorithm_feed_dict)

    values = sess.run(run_fetches, feed_dict)
    if summary_time:
      logging.info("Step #{}, {}".format(step, values["logging"]))
      self.summary_writer.add_summary(values["summaries"], step)
      self._last_summary_step = step

    if "fetches" in values:
      return step, values["fetches"]
    else:
      return step

  def train(self, algorithm, num_steps):
    global_step = tf.train.get_global_step()

    with self.managed_session() as sess:
      algorithm.start_training(sess, self.summary_writer,
                               self._summary_period)
      step = sess.run(global_step)
      # This will allow tensorboard to discard "orphaned" summaries when
      # reloading from checkpoint.
      purge_orphaned_summaries(self.summary_writer, step)
      last_summary_step = step - self._summary_period
      while not sess.should_stop() and step < num_steps:
        step = self.step(algorithm)


class SingularTrainer(DistributedTrainer):
  def __init__(self,
               logdir,
               summary_period,
               checkpoint_period,
               checkpoint=None,
               config=USE_DEFAULT):
    super(SingularTrainer, self).__init__(
        target='',
        is_chief=True,
        logdir=logdir,
        summary_period=summary_period,
        checkpoint_period=checkpoint_period,
        checkpoint=checkpoint,
        config=config)

  @contextmanager
  def managed_session(self, restore_vars=None, save_vars=None,
                      hooks=USE_DEFAULT):
    if self._checkpoint is not None:
      restorer = tf.train.Saver(restore_vars)
    if hooks == USE_DEFAULT:
      saver = tf.train.Saver(save_vars)
      hooks = [
          tf.train.CheckpointSaverHook(
              self._logdir,
              saver=saver,
              save_steps=self._checkpoint_period
            ),
      ]
    with tf.train.SingularMonitoredSession(hooks=hooks,
                                           config=self._config) as sess:
      if self._checkpoint is not None:
        restorer.restore(sess, self._checkpoint)
      self._session = sess
      yield sess
