import argparse
import gym
from gym.envs.atari import AtariEnv
import logging
import tensorflow as tf

import rl.policies as policies
import rl.trainers
from rl.trainers import A2CTrainer
import rl.wrappers

OPTIMIZER = tf.train.AdamOptimizer(1e-4)
# OPTIMIZER = tf.train.AdamOptimizer()
gym.undo_logger_setup()
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

def preprocess_wrap(env):
  if isinstance(env.unwrapped, AtariEnv):
    env = rl.wrappers.wrap(env, [
        rl.wrappers.UniverseStarterImageWrapper(),
        rl.wrappers.ClipRewardWrapper()
      ])
  env = rl.wrappers.LoggingWrapper()(env)
  return env


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
      "--env-id",
      required=True
  )
  parser.add_argument(
      "--logdir",
      required=True,
  )
  parser.add_argument(
      "--policy",
      required=True,
  )
  parser.add_argument(
      "--trajectory-length",
      type=int,
      default=20
  )
  parser.add_argument(
      "--entropy-coef",
      type=float,
      default=0.01
  )
  parser.add_argument(
      "--value-loss-coef",
      type=float,
      default=0.25
  )
  parser.add_argument(
      "--gamma",
      type=float,
      default=0.99
  )
  parser.add_argument(
      "--lambda_",
      type=float,
      default=1
  )
  parser.add_argument(
      "--num-train-steps",
      type=int,
      default=int(4 * 1e6)
  )
  parser.add_argument(
      "--summary-period",
      type=int,
      default=500
  )
  parser.add_argument(
      "--checkpoint-period",
      type=int,
      default=int(1e6)
  )
  parser.add_argument(
      "--checkpoint",
      default=None
  )
  args = parser.parse_args()
  return args

def main():
  args = get_args()

  env = preprocess_wrap(gym.make(args.env_id))
  policy_class = getattr(rl.policies, args.policy + "Policy")
  policy = policy_class(env.observation_space, env.action_space)
  logging.info("Using {} policy".format(policy_class))
  trainer = rl.trainers.A2CTrainer(env,
                       policy,
                       trajectory_length=args.trajectory_length,
                       entropy_coef=args.entropy_coef,
                       value_loss_coef=args.value_loss_coef)
  trainer.train(args.num_train_steps,
                OPTIMIZER,
                args.logdir,
                gamma=args.gamma,
                lambda_=args.lambda_,
                summary_period=args.summary_period,
                checkpoint_period=args.checkpoint_period,
                checkpoint=args.checkpoint)


if __name__ == "__main__":
  main()
