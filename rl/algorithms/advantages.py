import numpy as np

__all__ = ["GAE"]


class GAE(object):
  def __init__(self, policy, gamma=0.99, lambda_=0.95, normalize=False):
    self._policy = policy
    self._gamma = gamma
    self._lambda_ = lambda_
    self._normalize = normalize

  def __call__(self, trajectory, sess=None):
    num_timesteps = trajectory["num_timesteps"]
    gae = np.zeros([num_timesteps])
    gae[-1] = trajectory["rewards"][num_timesteps-1]\
        - trajectory["value_preds"][num_timesteps-1]
    if not trajectory["resets"][num_timesteps-1]:
      obs = trajectory["latest_observation"]
      gae[-1] += self._gamma * self._policy.act(obs, sess)[1]
    for i in reversed(range(num_timesteps-1)):
      not_reset = 1 - trajectory["resets"][i] # i is for next state
      delta = (
          trajectory["rewards"][i]
          + not_reset * self._gamma * trajectory["value_preds"][i+1]
          - trajectory["value_preds"][i]
      )
      gae[i] = delta + not_reset * self._gamma * self._lambda_ * gae[i+1]
    value_targets = gae + trajectory["value_preds"][:num_timesteps]
    if self._normalize:
      gae = (gae - gae.mean()) / (gae.std() + np.finfo(gae.dtype).eps)
    return gae, value_targets
