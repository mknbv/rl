import abc

from gym import spaces
import numpy as np
import tensorflow as tf


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
  def __init__(self, env, policy, batch_size, env_step=None):
    super(OnlineInteractionsProducer, self).__init__(env, policy, batch_size,
                                                     env_step=env_step)

  def start(self, session, summary_manager=None):
    super(OnlineInteractionsProducer, self).start(session, summary_manager)

    latest_observation = self._env.reset()
    obs_shape = (self._batch_size,) + latest_observation.shape
    obs_type = latest_observation.dtype
    if isinstance(self._env.action_space, spaces.Discrete):
      act_shape = tuple()
    else:
      act_shape = self._env.action_space.shape
    act_type = np.asarray(self.action_space.sample()).dtype
    act_shape = (self._batch_size,) + act_shape
    self._trajectory = {
        "latest_observation": latest_observation,
        "observations": np.empty(obs_shape, dtype=obs_type),
        "actions": np.empty(act_shape, dtype=act_type),
        "rewards": np.empty(self._batch_size, dtype=np.float32),
        "resets": np.empty(self._batch_size, dtype=np.bool),
        "critic_values": np.empty(self._batch_size, dtype=np.float32),
        "num_timesteps": self._batch_size,
    }
    if self._policy.state_values is not None:
      self._trajectory["policy_state"] = self._policy.state_values

  def next(self):
    traj = self._trajectory
    observations = traj["observations"]
    actions = traj["actions"]
    rewards = traj["rewards"]
    resets = traj["resets"]
    critic_values = traj["critic_values"]
    traj["num_timesteps"] = self._batch_size
    if "policy_state" in traj:
      traj["policy_state"] = self._policy.state_values
    for i in range(self._batch_size):
      observations[i] = self._trajectory["latest_observation"]
      actions[i], critic_values[i] =\
          self._policy.act(traj["latest_observation"], sess=self._sess)
      traj["latest_observation"], rewards[i], resets[i], info =\
          self._env.step(actions[i])
      if resets[i]:
        traj["latest_observation"] = self._env.reset()
        self._policy.reset()
        if self._summary_manager is not None:
          env_step = self._session.run(self.env_step) + i
          if self._summary_manager.summary_time(step=env_step):
            self._summary_manager.add_summary_dict(
                info.get("summaries", info), step=env_step)
        # Recurrent policies require trajectory to end when episode ends.
        # Otherwise the batch may combine interactions from differen episodes.
        if self._policy.state_inputs is not None:
          traj["num_timesteps"] = i + 1
          break

    self._update_env_step(traj["num_timesteps"])
    return self._trajectory
