from contextlib import contextmanager
import logging
import tensorflow as tf

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

  def _get_default_config(self):
    return tf.ConfigProto(intra_op_parallelism_threads=1,
                          inter_op_parallelism_threads=2)

  @contextmanager
  def managed_session(self):
    # Take only ps variables to save and check if initialization completed.
    ps_variables = list(filter(lambda v: v.device.startswith("/job:ps/"),
                               tf.global_variables()))
    ready_op = tf.report_uninitialized_variables(ps_variables)
    saver = tf.train.Saver(ps_variables)
    scaffold = tf.train.Scaffold(ready_for_local_init_op=ready_op,
                                 ready_op=ready_op, saver=saver)
    if self._is_chief:
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
      hooks = None
      session_creator = tf.train.WorkerSessionCreator(
          scaffold=scaffold, master=self._target, config=self._config)
    with tf.train.MonitoredSession(session_creator, hooks=hooks) as sess:
      yield sess

  def train(self, algorithm, num_steps):
    summary_writer = tf.summary.FileWriterCache.get(self._logdir)
    global_step = tf.train.get_global_step()

    with self.managed_session() as sess:
      algorithm.start_training(sess, summary_writer, self._summary_period)
      step = sess.run(global_step)
      # This will allow tensorboard to discard "orphaned" summaries when
      # reloading from checkpoint.
      summary_writer.add_session_log(
          tf.SessionLog(status=tf.SessionLog.START), step)
      last_summary_step = step - self._summary_period
      while not sess.should_stop() and step < num_steps:
        feed_dict = algorithm.get_feed_dict(sess)
        if not self._is_chief or\
            step - last_summary_step < self._summary_period:
          sess.run(algorithm.train_op, feed_dict)
        else:
          fetches = [
              algorithm.logging_fetches,
              algorithm.summaries,
              algorithm.train_op
          ]
          values, summaries = sess.run(fetches, feed_dict)[:-1]
          logging.info("Step #{}, {}".format(step, values))
          summary_writer.add_summary(summaries, step)
          last_summary_step = step
        step = sess.run(global_step)


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
  def managed_session(self):
    saver = tf.train.Saver()
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
        saver.restore(sess, self._checkpoint)
      yield sess
