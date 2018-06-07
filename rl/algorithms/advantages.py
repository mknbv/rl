import numpy as np


class ActorCriticAdvantage(object):
  def __init__(self, policy, gamma=0.99, normalize=False):
    self._policy = policy
    self._gamma = gamma
    self._normalize = normalize

  def __call__(self, trajectory, sess):
    value_targets = np.zeros_like(trajectory["critic_values"])
    num_envs = trajectory["latest_observations"].shape[0]
    value_targets[-num_envs:] = trajectory["rewards"][-num_envs:]
    obs = trajectory["latest_observations"]
    feed_dict = {self._policy.observations: obs}
    if self._policy.state_inputs is not None:
      feed_dict[self._policy.state_inputs] = self._policy.state_values
    last_value = sess.run(self._policy.critic_tensor, feed_dict)[:, 0]
    value_targets[-num_envs:] += (
        (1 - trajectory["resets"][-num_envs:]) * self._gamma * last_value)

    for i in range(value_targets.shape[0] - num_envs, 0, -num_envs):
      value_targets[i-num_envs:i] = (
          trajectory["rewards"][i-num_envs:i]
          + ((1 - trajectory["resets"][i-num_envs:i])
             * self._gamma * value_targets[i:i+num_envs])
      )

    advantages = value_targets[:,0] - trajectory["critic_values"][:,0]
    if self._normalize:
      advantages = (
          (advantages - advantages.mean())
          / (advantages.std() + np.finfo(advantages.dtype).eps)
      )
    return advantages, value_targets


class GAE(object):
  def __init__(self, policy, gamma=0.99, lambda_=0.95, normalize=False):
    self._policy = policy
    self._gamma = gamma
    self._lambda_ = lambda_
    self._normalize = normalize

  def __call__(self, trajectory, sess=None):
    gae = np.zeros(trajectory["env_steps"], dtype=np.float32)
    num_envs = trajectory["latest_observations"].shape[0]
    gae[-num_envs:] = (
        trajectory["rewards"][-num_envs:]
        - trajectory["critic_values"][-num_envs:,0]
    )
    obs = trajectory["latest_observations"]
    feed_dict = {self._policy.observations: obs}
    if self._policy.state_inputs is not None:
      feed_dict[self._policy.state_inputs] = self._policy.state_values
    last_values = sess.run(self._policy.critic_tensor, feed_dict)[:,0]
    gae[-num_envs:] += (
        (1 - trajectory["resets"][-num_envs:]) * self._gamma * last_values)

    for i in range(gae.shape[0] - num_envs, 0, -num_envs):
      not_reset = 1 - trajectory["resets"][i-num_envs:i]
      next_critic_values = trajectory["critic_values"][i:i+num_envs,0]
      delta = (
          trajectory["rewards"][i-num_envs:i]
          + not_reset * self._gamma * next_critic_values
          - trajectory["critic_values"][i-num_envs:i,0]
      )
      gae[i-num_envs:i] = (
          delta
          + not_reset * self._gamma * self._lambda_ * gae[i:i+num_envs]
      )
    value_targets = gae[:,None] + trajectory["critic_values"]

    if self._normalize:
      gae = (gae - gae.mean()) / (gae.std() + np.finfo(gae.dtype).eps)
    return gae, value_targets
