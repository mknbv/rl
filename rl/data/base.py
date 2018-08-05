import abc

import tensorflow as tf

from rl.training import SummaryManager


class BaseInteractionsProducer(abc.ABC):
  def __init__(self, env, policy, batch_size, env_step=None):
    self._env = env
    self._policy = policy
    self._batch_size = batch_size
    if env_step is None:
      env_step = tf.train.get_or_create_global_step()
    self._env_step = env_step
    self._elapsed_steps_ph = tf.placeholder(self._env_step.dtype, [],
                                            name="elapsed_steps")
    self._updated_env_step = self._env_step.assign_add(self._elapsed_steps_ph)
    self._summary_manager = None

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def observation_space(self):
    return self._env.observation_space

  @property
  def action_space(self):
    return self._env.action_space

  @property
  def env_step(self):
    return self._env_step

  @property
  def env_summary_manager(self):
    return self._env_summary_manager

  @abc.abstractmethod
  def start(self, session, env_summary_manager=None):
    if not self._policy.is_built:
      raise ValueError("Policy must be built before calling start")
    if not isinstance(env_summary_manager, EnvSummaryManager):
      raise TypeError(
          "env_summary_manager must be an instance of EnvSummaryManager, "
          "got object of type {}".format(type(env_summary_manager))
      )
    self._session = session
    self._env_summary_manager = env_summary_manager

  def _update_env_step(self, elapsed_steps):
    return self._session.run(self._updated_env_step,
                             {self._elapsed_steps_ph: elapsed_steps})

  @abc.abstractmethod
  def next(self):
    ...


class EnvSummaryManager(SummaryManager):
  def add_env_summary(self, info, step, session=None,
                      update_last_summary_step=True):
    self.add_summary_dict(info.get("summaries", info), step,
                          session=session,
                          update_last_summary_step=update_last_summary_step)


class EnvBatchSingleSummaryManager(EnvSummaryManager):
  def __init__(self, index=None, logdir=None, summary_writer=None,
               summary_period=None, last_summary_step=None):
    super(BatchEnvSingleSummaryManager, self).__init__(
        logdir=logdir,
        summary_writer=summary_writer,
        summary_period=summary_period,
        last_summary_step=last_summary_step)
    self._index = index

  def add_env_summary(self, infos, step, session=None):
    for index, info in infos:
      if index is None or index == self._index:
        super(EnvBatchSingleSummaryManager, self).add_env_summary(
            info, step, session=session)
        break


class EnvBatchAllSummaryManager(EnvSummaryManager):
  def __init__(self, batch_size, logdir=None, summary_writer=None,
               summary_period=None, last_summary_step=None):
    super(EnvBatchAllSummaryManager, self).__init__(
        logdir=logdir,
        summary_writer=summary_writer,
        summary_period=summary_period,
        last_summary_step=last_summary_step)
    self._batch_size = batch_size
    self._written_summaries = set()

  def add_env_summary(self, infos, step, session=None):
    for index, info in infos:
      if index not in self._written_summaries:
        self._written_summaries.add(index)
        all_written = len(self._written_summaries) == self._batch_size
        super(EnvBatchAllSummaryManager, self).add_env_summary(
            info, step, session=session, update_last_summary_step=all_written)
        if all_written:
          self._written_summaries.clear()
          break
