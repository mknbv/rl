import numpy as np


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
    last_values = sess.run(self._policy.critic_tensor,
                           {self._policy.observations: obs})[:,0]
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
