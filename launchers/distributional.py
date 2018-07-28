import argparse
import os
import pickle

import gym
import numpy as np
import rl
import tensorflow as tf
from tqdm import trange


def get_train_args(arg_list=None):
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--logdir", required=True)
  parser.add_argument("--checkpoint")
  parser.add_argument("--experience")
  parser.add_argument("--train-seed", type=int, default=0)
  parser.add_argument("--summary-period", type=int, default=int(1e4))
  parser.add_argument("--kind", choices=["qr-dqn-0", "qr-dqn-1", "dqn"],
                      default="qr-dqn-1")
  args = parser.parse_args(arg_list)
  return args

def get_perform_args(arg_list=None):
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--seeds", type=int, nargs="+", default=[1])
  parser.add_argument("--num-episodes", type=int, default=1)
  parser.add_argument("--render", action="store_true")
  parser.add_argument("--epsilon", type=float, default=1e-3)
  args = parser.parse_args(arg_list)
  return args


def evaluate(env_id, policy, summary_writer, seeds=[0], epsilon=1e-3,
             nframes=int(5e5), step=None, sess=None):
  if sess is None:
    sess = tf.get_default_session()
  if step is None:
    step = sess.run(tf.train.get_global_step())
  summary = tf.Summary()

  rl.logger.info("Evaluation starts, step #{}".format(step))
  env = gym.make(env_id)
  env = rl.env_wrappers.nature_dqn_wrap(env, clip_reward=False)
  for seed in seeds:
    env.seed(seed)
    obs = env.reset()
    rewards = [0]
    episode_lengths = [0]
    for i in trange(0, nframes, 4):
      action = policy.act(obs[None], epsilon_value=epsilon,
                          sess=sess)["actions"][0]
      obs, rew, done, info = env.step(action)
      rewards[-1] += rew
      episode_lengths[-1] += 1
      if done:
        env.reset()
      if ("real_done" in info and info["real_done"])\
          or ("real_done" not in info and done):
        rewards.append(0)
        episode_lengths.append(0)

  num_episodes = len(rewards) - 1
  if not done and len(rewards) > 1:
    rewards = rewards[:-1]
    episode_lengths = episode_lengths[:-1]

  rl.logger.info("Evaluation finished, step #{}, mean reward: {}"
               .format(step, np.mean(rewards)))
  summary.value.add(tag="Evaluation/reward_mean", simple_value=np.mean(rewards))
  summary.value.add(tag="Evaluation/reward_median",
                    simple_value=np.median(rewards))
  summary.value.add(tag="Evaluation/min_reward", simple_value=np.min(rewards))
  summary.value.add(tag="Evaluation/max_reward", simple_value=np.max(rewards))
  summary.value.add(tag="Evaluation/num_episodes", simple_value=num_episodes)
  summary.value.add(tag="Evaluation/reward_std", simple_value=np.std(rewards))
  summary.value.add(tag="Evaluation/mean_episode_length",
                    simple_value=np.mean(episode_lengths))
  summary_writer.add_summary(summary, step)

  return rewards


def train(env_id, arg_list=None):
  args = get_train_args(arg_list)

  env = gym.make(env_id)
  env = rl.env_wrappers.SummariesInfo(env)
  env = rl.env_wrappers.nature_dqn_wrap(env)
  env.seed(args.train_seed)
  global_step = tf.train.create_global_step()
  epsilon = tf.train.polynomial_decay(
      1.0,
      global_step,
      decay_steps=int(1e6+1e5) // 4,
      end_learning_rate=0.01,
      name="epsilon_decay"
  )

  policy = rl.policies.DistributionalPolicy(
      env.observation_space,
      env.action_space,
      core=rl.policies.NatureDQNCore(),
      epsilon=epsilon,
      nbins=1 if args.kind == "dqn" else 200,
      ubyte_rescale=True
  )

  experience_replay = rl.data.UniformExperienceReplay(
      env, policy,
      experience_size=int(1e6),
      experience_start_size=int(5e4),
      batch_size=32
  )
  if args.experience is not None:
    experience_replay.restore_experience(args.experience)

  if args.kind == "dqn":
    algorithm = rl.algorithms.DQNAlgorithm(
        experience_replay=experience_replay,
        policy=policy,
        target_update_period=4 * int(1e4),
    )
  else:
    algorithm = rl.algorithms.DistributionalAlgorithm(
        experience_replay=experience_replay,
        policy=policy,
        target_update_period=4 * int(1e4),
        kind=args.kind
    )

  summary_manager = rl.training.SummaryManager(
      logdir=args.logdir, summary_period=args.summary_period)
  trainer = rl.training.SingularTrainer(
      summary_manager=summary_manager,
      checkpoint_period=int(1e6),
      checkpoint=args.checkpoint
  )

  if args.kind == "dqn":
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=0.00025,
        decay=0.95,
        momentum=0.0,
        epsilon=0.01
    )
  else:
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.00005,
        epsilon=0.01/32,
    )
  algorithm.build(optimizer)
  inc_step = tf.train.get_global_step().assign_add(4)

  best_model_saver = tf.train.Saver(max_to_keep=1)
  with tf.variable_scope("evaluation"):
    best_reward = tf.get_variable(
        "best_reward", shape=[], dtype=tf.float32,
        initializer=tf.constant_initializer(-np.inf)
    )
  running_model_saver = tf.train.Saver(max_to_keep=None)

  with trainer.managed_session(hooks=None) as sess:
    experience_replay.start(sess, summary_manager.copy())

    step = sess.run(global_step)
    last_checkpoint_step = None if step == 0 else step - step % int(1e6)
    last_evaluation_step = None if step == 0 else step - step % int(1e6)
    while not sess.should_stop() and step <= int(200 * 1e6):
      step, _ = trainer.step(algorithm=algorithm, fetches=[inc_step])
      if (last_checkpoint_step is None\
          or step - last_checkpoint_step >= int(1e6))\
          and experience_replay.storage.latest()["reset"]:
        running_model_saver.save(sess.raw_session(),
                                 os.path.join(args.logdir, "model.ckpt"),
                                 global_step=step)
        experience_replay.storage.save(
            os.path.join(args.logdir, "experience.pickle"))
        last_checkpoint_step = step - step % int(1e6)
      if last_evaluation_step is None\
          or step - last_evaluation_step >= int(1e6):
        mean_policy_reward = np.mean(
            evaluate(env.spec.id, policy, summary_manager.summary_writer,
                     step=step, sess=sess)
        )
        if mean_policy_reward > best_reward.eval(sess):
          best_model_saver.save(sess.raw_session(),
                                os.path.join(args.logdir, "best-model.ckpt"),
                                global_step=step)
          best_reward.load(mean_policy_reward, session=sess)
        last_evaluation_step = step


class RenderEverything(gym.Wrapper):
  def __init__(self, env, mode="human"):
    super(RenderEverything, self).__init__(env)
    self._mode = mode

  def step(self, action):
    ret = self.env.step(action)
    self.env.render(mode=self._mode)
    return ret

  def reset(self):
    obs = self.env.reset()
    self.env.render(mode=self._mode)
    return obs


def perform(env_id, arg_list=None):
  args = get_perform_args(arg_list)
  env = gym.make(env_id)
  if args.render:
    env = RenderEverything(env)
  env = rl.env_wrappers.nature_dqn_wrap(env, clip_reward=False)

  policy = rl.policies.DistributionalPolicy(
      env.observation_space,
      env.action_space,
      core=rl.policies.DQNCore(kind="nature"),
      epsilon=tf.constant(args.epsilon),
      nbins=200
  )
  policy.build()
  saver = tf.train.Saver(policy.var_list())

  all_rewards = []
  rewards = []
  with tf.Session() as sess:
    saver.restore(sess, args.checkpoint)

    for seed in args.seeds:
      env.seed(seed)
      for j in range(args.num_episodes):
        obs = env.reset()
        done = False
        rewards.append(0)
        while not done:
          obs, rew, done, info = env.step(policy.act(obs))
          all_rewards.append(rew)
          rewards[-1] += rew
          if "real_done" in info:
            done = info["real_done"]
        print(
          "[{}/{}] reward: {} mean reward: {}, std: {}"
          .format(j+1, args.num_episodes, rewards[-1],
                  np.mean(rewards), np.std(rewards))
        )


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--env-id", required=True)
  parser.add_argument("--perform", action="store_true")
  args, unknown = parser.parse_known_args()
  if not args.perform:
    train(args.env_id, unknown)
  else:
    perform(args.env_id, unknown)


if __name__ == "__main__":
  main()
