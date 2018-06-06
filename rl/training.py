from copy import copy
from contextlib import contextmanager
from logging import getLogger

import tensorflow as tf

from rl.utils.tf_utils import (purge_orphaned_summaries
                               as _purge_orphaned_summaries)

USE_DEFAULT = object()
logger = getLogger("rl")


class SummaryManager(object):
  def __init__(self, logdir=None, summary_writer=None, summary_period=None,
               last_summary_step=None):
    if (logdir is None) == (summary_writer is None):
      raise ValueError("exactly one of logdir or summary_writer must be set")

    if summary_writer is None:
      summary_writer = tf.summary.FileWriterCache.get(logdir)
    self._summary_writer = summary_writer
    self._summary_period = summary_period
    self._last_summary_step = last_summary_step

  @property
  def summary_writer(self):
    return self._summary_writer

  def copy(self):
    return copy(self)

  def _get_step(self, step, session=None):
    if isinstance(step, (tf.Variable, tf.Tensor)):
      if session is None:
        raise ValueError("session is None when step is instance %s"
                         % type(step))
      step = session.run(step)
    return step

  def summary_time(self, step, session=None):
    step = self._get_step(step, session)
    if self._summary_period is None:
      return False
    elif self._last_summary_step is None:
      return True
    else:
      return step - self._last_summary_step >= self._summary_period

  def add_summary(self, summary, step, session=None,
                  update_last_summary_step=True):
    step = self._get_step(step, session)
    if step is None:
      step = session.run(self._step)
    self._summary_writer.add_summary(summary, global_step=step)
    if update_last_summary_step:
      self._last_summary_step = step

  def add_summary_dict(self, summary_dict, step, session=None,
                       update_last_summary_step=True):
    summary = tf.Summary()
    for key, val in summary_dict.items():
      summary.value.add(tag=key, simple_value=val)
    self.add_summary(summary, step=step, session=session,
                     update_last_summary_step=update_last_summary_step)


class DistributedTrainer(object):
  def __init__(self,
               target,
               is_chief,
               summary_manager=None,
               checkpoint_dir=None,
               checkpoint_period=None,
               checkpoint=None,
               config=USE_DEFAULT):
    self._target = target
    self._is_chief = is_chief
    self._summary_manager = summary_manager
    if (summary_manager is None
        and checkpoint_dir is None
        and checkpoint_period is not None):
      raise ValueError("Either summary_manager or checkpoint_dir must be"
                       " specified when checkpoint_period is not None")
    if checkpoint_dir is not None and checkpoint_period is None:
      raise ValueError("checkpoint_period must be specified"
                       " when checkpoint_dir is not None")

    if checkpoint_period is not None and checkpoint_dir is None:
      checkpoint_dir = summary_manager.summary_writer.get_logdir()

    self._checkpoint_dir = checkpoint_dir
    self._checkpoint_period = checkpoint_period
    self._checkpoint = checkpoint
    if config == USE_DEFAULT:
      config = self._get_default_config()
    self._config = config
    self._session = None

  def _get_default_config(self):
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=2)
    # Dynamic memory allocation on gpu.
    # https://github.com/tensorflow/tensorflow/issues/1578
    config.gpu_options.allow_growth = True
    return config

  @contextmanager
  def managed_session(self, init_vars=None, restore_vars=None,
                      save_vars=None, hooks=USE_DEFAULT,
                      purge_orphaned_summaries=True):
    if init_vars is None:
      init_vars = list(filter(
        lambda v: v.device.startswith("/job:ps/"),
        tf.global_variables()
      ))
    if restore_vars is None:
      restore_vars = init_vars
    if save_vars is None:
      save_vars = init_vars
    ready_op = tf.report_uninitialized_variables(init_vars)
    restorer = tf.train.Saver(restore_vars)
    scaffold = tf.train.Scaffold(ready_for_local_init_op=ready_op,
                                 ready_op=ready_op, saver=restorer)

    if hooks == USE_DEFAULT:
      if self._is_chief and self._checkpoint_dir is not None:
        saver = tf.train.Saver(save_vars)
        hooks = [
            tf.train.CheckpointSaverHook(
              checkpoint_dir=self._checkpoint_dir, saver=saver,
              save_steps=self._checkpoint_period)
        ]
      else:
        hooks = None

    if self._is_chief:
      session_creator = tf.train.ChiefSessionCreator(
          scaffold=scaffold, master=self._target,
          config=self._config, checkpoint_filename_with_path=self._checkpoint)
    else:
      session_creator = tf.train.WorkerSessionCreator(
          scaffold=scaffold, master=self._target, config=self._config)

    with tf.train.MonitoredSession(session_creator, hooks=hooks) as sess:
      if purge_orphaned_summaries and self._summary_manager is not None:
        _purge_orphaned_summaries(self._summary_manager.summary_writer,
                                  sess.run(tf.train.get_global_step()))
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
        and self._summary_manager is not None
        and self._summary_manager.summary_time(step=step)):
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
      logger.info("Step #{}, {}".format(step, values["logging"]))
      self._summary_manager.add_summary(values["summaries"], step=step)

    if "fetches" in values:
      return step, values["fetches"]
    else:
      return step

  def train(self, algorithm, num_steps):
    global_step = tf.train.get_global_step()

    def _train(sess):
      step = sess.run(global_step)
      while not sess.should_stop() and step < num_steps:
        step = self.step(algorithm)

    if self._session is not None:
      _train(self._session)
    else:
      with self.managed_session() as sess:
        _train(sess)


class SingularTrainer(DistributedTrainer):
  def __init__(self,
               summary_manager=None,
               checkpoint_dir=None,
               checkpoint_period=None,
               checkpoint=None,
               config=USE_DEFAULT):
    super(SingularTrainer, self).__init__(
        target='',
        is_chief=True,
        summary_manager=summary_manager,
        checkpoint_dir=checkpoint_dir,
        checkpoint_period=checkpoint_period,
        checkpoint=checkpoint,
        config=config)

  @contextmanager
  def managed_session(self, save_vars=None, restore_vars=None,
                      hooks=USE_DEFAULT, purge_orphaned_summaries=True):
    if self._checkpoint is not None:
      restorer = tf.train.Saver(restore_vars)
    if hooks == USE_DEFAULT:
      if self._checkpoint_dir is not None:
        saver = tf.train.Saver(save_vars)
        hooks = [
            tf.train.CheckpointSaverHook(
                self._checkpoint_dir,
                saver=saver,
                save_steps=self._checkpoint_period
              ),
        ]
      else:
        hooks = None
    with tf.train.SingularMonitoredSession(hooks=hooks,
                                           config=self._config) as sess:
      if self._checkpoint is not None:
        restorer.restore(sess, self._checkpoint)
      if purge_orphaned_summaries and self._summary_manager is not None:
        _purge_orphaned_summaries(self._summary_manager.summary_writer,
                                  sess.run(tf.train.get_global_step()))
      self._session = sess
      yield sess
