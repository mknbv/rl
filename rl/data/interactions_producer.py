import abc

import numpy as np
import tensorflow as tf

from rl.utils.env_batch import EnvBatch, SingleEnvBatch


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
  def observation_space(self):
    return self._env.observation_space

  @property
  def action_space(self):
    return self._env.action_space

  @property
  def env_step(self):
    return self._env_step

  @property
  def summary_manager(self):
    return self._summary_manager

  @abc.abstractmethod
  def start(self, session, summary_manager=None):
    if not self._policy.is_built:
      raise ValueError("Policy must be built before calling start")
    self._session = session
    self._summary_manager = summary_manager

  def _update_env_step(self, elapsed_steps):
    return self._session.run(self._updated_env_step,
                             {self._elapsed_steps_ph: elapsed_steps})

  @abc.abstractmethod
  def next(self):
    ...


class OnlineInteractionsProducer(BaseInteractionsProducer):
  def __init__(self, env, policy, batch_size, cutoff=True, env_step=None):
    if not isinstance(env, EnvBatch):
      env = SingleEnvBatch(env)
    if batch_size % env.num_envs != 0:
      raise ValueError("env.num_envs = {} does not divide batch_size = {}"
                       .format(env.num_envs, batch_size))
    super(OnlineInteractionsProducer, self).__init__(env, policy, batch_size,
                                                     env_step=env_step)
    self._cutoff = cutoff

  def start(self, session, summary_manager=None):
    super(OnlineInteractionsProducer, self).start(session, summary_manager)

    latest_observations = np.array(self._env.reset())
    obs_shape = (self._batch_size,) + latest_observations.shape[1:]
    obs_type = latest_observations.dtype
    self._trajectory = {
        "latest_observations": latest_observations,
        "observations": np.empty(obs_shape, dtype=obs_type),
        "rewards": np.empty(self._batch_size, dtype=np.float32),
        "resets": np.empty(self._batch_size, dtype=np.bool),
    }
    act = self._policy.act(latest_observations, sess=session)
    for key, val in act.items():
      val_batch_shape = (self._batch_size,) + val.shape[1:]
      self._trajectory[key] = np.empty(val_batch_shape, val.dtype)

  def next(self):
    observations = self._trajectory["observations"]
    actions = self._trajectory["actions"]
    self._trajectory["env_steps"] = self._batch_size
    if "policy_state" in self._trajectory:
      self._trajectory["policy_state"] = self._policy.state_values
    n = self._env.num_envs

    for i in range(0, self._batch_size, n):
      batch_slice = slice(i, i + n)
      observations[batch_slice] = self._trajectory["latest_observations"]
      act = self._policy.act(self._trajectory["latest_observations"],
                             sess=self._session)
      for key, val in act.items():
        self._trajectory[key][batch_slice] = val
      obs, rews, resets, infos = self._env.step(actions[batch_slice])
      self._trajectory["latest_observations"] = obs
      self._trajectory["rewards"][batch_slice] = rews
      self._trajectory["resets"][batch_slice] = resets

      if np.any(resets):
        self._policy.reset(resets)
        if self.summary_manager is not None:
          env_step = self._session.run(self.env_step) + i + n
          if self.summary_manager.summary_time(env_step):
            for info in np.asarray(infos)[resets]:
              self.summary_manager.add_summary_dict(
                  info.get("summaries", info), step=env_step)

        if self._cutoff:
          self._trajectory["env_steps"] = i + n
          break

    self._update_env_step(self._trajectory["env_steps"])
    if self._trajectory["env_steps"] == self._batch_size:
      return self._trajectory
    else:
      traj = {}
      for key, val in self._trajectory.items():
        if key != "latest_observation" and isinstance(val, np.ndarray):
          traj[key] = val[:self._trajectory["env_steps"]]
        else:
          traj[key] = val
      return traj
