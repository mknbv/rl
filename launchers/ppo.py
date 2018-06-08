import argparse
from math import sqrt

import gym
import rl
import tensorflow as tf


def get_parser(parser=None):
  if parser is None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--env-id", required=True)
  parser.add_argument("--logdir", required=True)
  parser.add_argument("--num-timesteps", type=int, default=int(10e6))
  parser.add_argument("--num-epochs", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=128 * 8)
  parser.add_argument("--num-minibatches", type=int, default=4)
  parser.add_argument("--summary-period", type=int, default=int(1e3))
  parser.add_argument("--checkpoint-period", type=int, default=int(1e6))
  parser.add_argument("--checkpoint")
  return parser


def make_env(env_id, seed=0, rank=0, clip_reward=True):
  env = gym.make(env_id)
  env.seed(seed + rank)
  env = rl.env_wrappers.SummariesInfo(env)
  env = rl.env_wrappers.nature_dqn_wrap(env, lazy=False,
                                        clip_reward=clip_reward)
  return env


def main():
  args = get_parser().parse_args()

  env = rl.env_batch.ParallelEnvBatch([
      lambda: make_env(args.env_id, rank=i, clip_reward=True)
      for i in range(8)
  ])

  ppo_step = tf.get_variable("ppo_step", [], tf.int32, trainable=False,
                             initializer=tf.zeros_initializer())

  optimizer = tf.train.AdamOptimizer(
      learning_rate=2.5e-4,
      epsilon=1e-5)

  policy = rl.policies.CategoricalActorCriticPolicy(
      env.observation_space,
      env.action_space,
      core=rl.policies.NatureDQNCore(
        kernel_initializer=rl.tf_utils.orthogonal_initializer(sqrt(2)),
        bias_initializer=tf.zeros_initializer()
      ),
      compute_log_prob=True,
  )
  interactions_producer = rl.data.OnlineInteractionsProducer(
      env, policy,
      batch_size=args.batch_size,
      cutoff=False,
  )
  cliprange = tf.train.polynomial_decay(
      0.1,
      global_step=interactions_producer.env_step,
      decay_steps=1.1 * args.num_timesteps,
      end_learning_rate=0)

  ppo = rl.algorithms.PPO2Algorithm(
      interactions_producer,
      policy,
      num_epochs=args.num_epochs,
      num_minibatches=args.num_minibatches,
      cliprange=cliprange,
      max_grad_norm=0.5,
  )
  ppo.build(optimizer)

  ppo_summary_manager = rl.training.SummaryManager(
      logdir=args.logdir, summary_period=args.summary_period)
  trainer = rl.training.SingularTrainer(
      checkpoint_dir=args.logdir,
      checkpoint_period=int(1e6),
      checkpoint=args.checkpoint
  )

  inc_ppo_step = ppo_step.assign_add(1)
  with trainer.managed_session() as sess:
    env_step = sess.run(interactions_producer.env_step)
    env_summary_manager = ppo_summary_manager.copy()
    interactions_producer.start(sess, env_summary_manager)

    while not sess.should_stop() and env_step <= args.num_timesteps:
      fetches = {"ppo_step": inc_ppo_step}
      if ppo_summary_manager.summary_time(step=env_step):
        fetches["summaries"] = ppo.summaries
        fetches["logging"] = ppo.logging_fetches
      env_step, values = trainer.step(ppo, fetches=fetches)
      if "summaries" in values:
        rl.logger.info(values["logging"])
        ppo_summary_manager.summary_writer.add_summary(
            values["summaries"], global_step=values["ppo_step"])
        ppo_summary_manager.update_last_summary_step(env_step)


if __name__ == "__main__":
  main()
